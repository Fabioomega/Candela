use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use candela::backend::{Backend, ComputeFor, DefaultBackend};
use candela::{BakedPromise, Composable, Dimension, Layout, OpError, Skeleton, Tensor};

trait EvictionPolicy {
    /// The constructor of the policy
    ///
    /// Creates a policy that must manage at least `cache_size` items.
    /// The cache may grow depending on the policy eviction behavior.
    fn new(cache_size: usize) -> Self;

    /// The get action
    ///
    /// Is called when an element is being read. This element is guaranteed
    /// to exist in the cache.
    fn on_get(&mut self, idx: usize);

    /// The insert action
    ///
    /// Is called when an element is being inserted. The element is guaranteed
    /// to not exist in the cache.
    fn on_insert(&mut self, idx: usize);

    /// The remove action
    ///
    /// Is called when an element is being removed. The element is guaranteed
    /// to exist in the cache.
    fn on_remove(&mut self, idx: usize);

    /// The eviction action
    ///
    /// Is called when a new element must be added to the cache and it does not
    /// have enough space in the current arena.
    /// Returning `None` means that the arena should grow to accommodate the new element
    /// instead of removing an element, while `Some(idx)` means that the idx in the arena
    /// should be used instead.
    fn evict(&mut self) -> Option<usize>;
}

//////////////////////////////////////////////////////////////

struct UnboundedPolicy;

impl EvictionPolicy for UnboundedPolicy {
    fn new(_: usize) -> Self {
        Self {}
    }

    fn on_get(&mut self, _: usize) {}

    fn on_insert(&mut self, _: usize) {}

    fn on_remove(&mut self, _: usize) {}

    fn evict(&mut self) -> Option<usize> {
        None
    }
}

//////////////////////////////////////////////////////////////

struct Slot<Key, T, B: Backend> {
    key: Key,
    sk: Arc<Skeleton<T, B>>,
}

impl<Key: Clone, T, B: Backend> Clone for Slot<Key, T, B> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            sk: self.sk.clone(),
        }
    }
}

struct Cache<Key: Clone + Hash, T, B: Backend> {
    arena: Vec<Option<Slot<Key, T, B>>>,
    map: HashMap<Key, usize>,
}

struct Link {
    prev: usize,
    next: usize,
}

struct LRUPolicy {
    order: HashMap<usize, Link>,
    head: usize,
    tail: usize,
}

impl EvictionPolicy for LRUPolicy {
    fn new(cache_size: usize) -> Self {
        Self {
            order: HashMap::with_capacity(cache_size),
            head: usize::MAX,
            tail: usize::MAX,
        }
    }

    // The caller must ensure that the idx exist!
    // Assumes that at least a single value is present (head and tail != usize::MAX)
    fn on_get(&mut self, idx: usize) {
        if idx == self.head {
            return;
        }

        let [Some(recent), Some(head)] = self.order.get_disjoint_mut([&idx, &self.head]) else {
            unreachable!("on_get should only be called if we are sure that the idx exists")
        };

        let recent_next = recent.next;
        let recent_previous = recent.prev;

        recent.next = self.head;
        recent.prev = usize::MAX;
        head.prev = idx;

        if idx != self.tail {
            let [Some(previous), Some(next)] = self
                .order
                .get_disjoint_mut([&recent_previous, &recent_next])
            else {
                unreachable!("on_get should only be called if we are sure that the idx exists")
            };

            previous.next = recent_next;
            next.prev = recent_previous;
        } else {
            let previous = self.order.get_mut(&recent_previous).unwrap();

            previous.next = usize::MAX;
            self.tail = recent_previous;
        }

        self.head = idx;
    }

    fn on_insert(&mut self, idx: usize) {
        self.order.insert(
            idx,
            Link {
                prev: usize::MAX,
                next: self.head,
            },
        );

        if self.head != usize::MAX {
            let older_head = self.order.get_mut(&self.head).unwrap();
            older_head.prev = idx;
        }

        self.head = idx;

        if self.tail == usize::MAX {
            self.tail = idx;
        }
    }

    fn on_remove(&mut self, idx: usize) {
        if idx == self.head {
            let next = self.order.remove(&idx).unwrap().next;

            if next != usize::MAX {
                self.order.get_mut(&next).unwrap().prev = usize::MAX;
            } else {
                self.tail = usize::MAX;
            }

            self.head = next;
        } else if idx == self.tail {
            let previous = self.order.remove(&idx).unwrap().prev;

            if previous != usize::MAX {
                self.order.get_mut(&previous).unwrap().next = usize::MAX;
            }

            self.tail = previous;
        } else {
            let recent = self.order.remove(&idx).unwrap();

            let [Some(previous), Some(next)] =
                self.order.get_disjoint_mut([&recent.prev, &recent.next])
            else {
                panic!()
            };

            previous.next = recent.next;
            next.prev = recent.prev;
        }
    }

    // Assumes that at least a single element is present
    fn evict(&mut self) -> Option<usize> {
        let tail_idx = self.tail;

        let tail = self.order.remove(&tail_idx).unwrap();

        if tail.prev != usize::MAX {
            self.order.get_mut(&tail.prev).unwrap().next = usize::MAX;
        } else {
            self.head = usize::MAX;
        }

        self.tail = tail.prev;

        Some(tail_idx)
    }
}

//////////////////////////////////////////////////////////////

struct SkeletonCacheInner<Key: Clone + Hash + Eq, P: EvictionPolicy, T, B: Backend> {
    cache: Cache<Key, T, B>,
    free: Vec<usize>,
    policy: P,
}

struct SkeletonCache<Key: Clone + Hash + Eq, P: EvictionPolicy, T, B: Backend = DefaultBackend>(
    Mutex<SkeletonCacheInner<Key, P, T, B>>,
);

impl<Key, P: EvictionPolicy, T, B> SkeletonCache<Key, P, T, B>
where
    Key: Clone + Hash + Eq,
    T: Clone + PartialEq + ComputeFor<B>,
    B: Backend,
{
    fn insert_pair(
        &self,
        mut lock: std::sync::MutexGuard<'_, SkeletonCacheInner<Key, P, T, B>>,
        key: Key,
        value: Arc<Skeleton<T, B>>,
    ) {
        if let Some(idx) = lock.free.pop() {
            lock.cache.arena[idx] = Some(Slot {
                key: key.clone(),
                sk: value,
            });

            lock.cache.map.insert(key, idx);
            lock.policy.on_insert(idx);
        } else {
            let idx = lock.policy.evict();

            match idx {
                Some(idx) => {
                    let slot = lock.cache.arena[idx]
                        .replace(Slot {
                            key: key.clone(),
                            sk: value,
                        })
                        .unwrap();

                    lock.cache.map.remove(&slot.key);
                    lock.cache.map.insert(key, idx);
                    lock.policy.on_insert(idx);
                }
                None => {
                    let idx = lock.cache.arena.len();

                    lock.cache.arena.push(Some(Slot {
                        key: key.clone(),
                        sk: value,
                    }));

                    lock.cache.map.insert(key, idx);
                    lock.policy.on_insert(idx);
                }
            }
        }
    }

    pub fn new(cache_size: usize) -> Self {
        let mut v: Vec<Option<Slot<Key, T, B>>> = Vec::with_capacity(cache_size);
        v.resize(cache_size, None);

        let free: Vec<usize> = (0..cache_size).collect();

        Self(Mutex::new(SkeletonCacheInner {
            cache: Cache {
                arena: v,
                map: HashMap::with_capacity(cache_size),
            },
            free,
            policy: P::new(cache_size),
        }))
    }

    pub fn get_or_insert_with<F>(&self, key: &Key, build: F) -> Arc<Skeleton<T, B>>
    where
        F: FnOnce() -> Skeleton<T, B>,
    {
        let mut skeleton: Option<Arc<Skeleton<T, B>>> = None;

        {
            let mut lock = self.0.lock().unwrap();

            if let Some(&idx) = lock.cache.map.get(key) {
                skeleton = Some(lock.cache.arena[idx].as_ref().unwrap().sk.clone());
                lock.policy.on_get(idx);
            }
        }

        if let Some(sk) = skeleton {
            return sk;
        }

        let sk: Arc<Skeleton<T, B>> = Arc::new(build());

        {
            let lock = self.0.lock().unwrap();

            if !lock.cache.map.contains_key(key) {
                self.insert_pair(lock, key.clone(), sk.clone());
            }
        }

        sk
    }

    pub fn remove(&self, key: &Key) -> Option<Arc<Skeleton<T, B>>> {
        let mut lock = self.0.lock().unwrap();

        let idx = lock.cache.map.remove(key)?;
        let slot = lock.cache.arena[idx].take();
        lock.free.push(idx);
        lock.policy.on_remove(idx);

        Some(slot.unwrap().sk)
    }

    pub fn contains_key(&self, key: &Key) -> bool {
        let lock = self.0.lock().unwrap();

        lock.cache.map.contains_key(key)
    }
}

type BuildFunction<T, B> = Box<dyn Fn(&[Layout]) -> Skeleton<T, B>>;

impl<P, T, B> SkeletonCache<Box<[Layout]>, P, T, B>
where
    P: EvictionPolicy,
    T: Clone + PartialEq + ComputeFor<B>,
    B: Backend,
{
    pub fn run(
        &self,
        inputs: &[&Tensor<T, B>],
        on_miss: &BuildFunction<T, B>,
    ) -> Result<Tensor<T, B>, OpError> {
        let input_layouts: Box<[Layout]> = inputs.iter().map(|&t| t.layout().clone()).collect();

        let sk: Arc<Skeleton<T, B>> =
            self.get_or_insert_with(&input_layouts, || on_miss(&input_layouts));
        sk.run(inputs)
    }

    pub fn compose<C>(
        &self,
        inputs: &[&C],
        on_miss: &BuildFunction<T, B>,
    ) -> Result<BakedPromise<T, B>, OpError>
    where
        C: Composable<T, B>,
    {
        let input_layouts: Box<[Layout]> = inputs.iter().map(|&t| t.layout().clone()).collect();

        let sk: Arc<Skeleton<T, B>> =
            self.get_or_insert_with(&input_layouts, || on_miss(&input_layouts));
        sk.compose(inputs)
    }
}

//////////////////////////////////////////////////////////////

/// A cache (group) of skeletons with different shapes
///
/// This is a hashmap abstraction on top of a [`Skeleton`] to enable dynamic shapes.
/// It calls the [`BuildFunction`] every time a group of tensors with never-before-seen
/// layouts arrives, and stores the result in the cache. The cache size and eviction
/// behaviour are determined by the chosen policy, which must implement [`EvictionPolicy`].
///
/// The build function must bind its slots in the same order as the layouts it receives,
/// returning a [`Skeleton`] that supports that shape.
struct DynamicSkeleton<T, B: Backend = DefaultBackend, P: EvictionPolicy = LRUPolicy> {
    cache: SkeletonCache<Box<[Layout]>, P, T, B>,
    build: BuildFunction<T, B>,
}

impl<P: EvictionPolicy, T, B: Backend> DynamicSkeleton<T, B, P>
where
    T: ComputeFor<B>,
    B: Backend,
{
    /// Creates a new dynamic skeleton
    ///
    /// Creates a cache of at least `cache_size` items, where each entry maps a
    /// `Layout` to a [`Skeleton`].
    ///
    /// On a miss it calls `build` to create and cache a new skeleton; the slots bound
    /// in `build` must be in the same order as its `inputs` argument.
    ///
    /// # Examples
    /// ```
    /// use candela::{DynamicSkeleton, Layout, Skeleton, SkeletonSlot, Tensor};
    /// use std::error::Error;
    ///
    /// fn build(inputs: &[Layout]) -> Skeleton<f32> {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// }
    ///
    /// fn main() -> Result<(), Box<dyn Error>> {
    ///     let a = Tensor::from_scalar(0.3, &[4]);
    ///     let b = Tensor::from_scalar(0.3, &[8]);
    ///
    ///     let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(12, Box::new(build));
    ///     let out_a = sk.run(&[&a])?;
    ///     let out_b = sk.run(&[&b])?;
    ///
    ///     println!("{out_a}");
    ///     println!("{out_b}");
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn new(cache_size: usize, build: BuildFunction<T, B>) -> Self {
        Self {
            cache: SkeletonCache::new(cache_size),
            build,
        }
    }

    pub fn run(&self, inputs: &[&Tensor<T, B>]) -> Result<Tensor<T, B>, OpError> {
        self.cache.run(inputs, &self.build)
    }

    pub fn compose<C>(&self, inputs: &[&C]) -> Result<BakedPromise<T, B>, OpError>
    where
        C: Composable<T, B>,
    {
        self.cache.compose(inputs, &self.build)
    }

    // TODO: Is contains_key, remove, insert and get useful here?
    // If a usecase ever presents itself we can add them here.
}

type UnboundedDynamicSkeleton<T, B> = DynamicSkeleton<T, B, UnboundedPolicy>;

//////////////////////////////////////////////////////////////

use candela::SkeletonSlot;
use std::error::Error;

fn build(inputs: &[Layout]) -> Skeleton<f32> {
    let a = SkeletonSlot::new(inputs[0].clone());
    (&a * 2.0).into_skeleton(&[a]).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let a = Tensor::from_scalar(0.3, &[4]);
    let b = Tensor::from_scalar(0.3, &[8]);

    let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(12, Box::new(build));
    let out_a = sk.run(&[&a])?;
    let out_b = sk.run(&[&b])?;

    println!("{out_a}");
    println!("{out_b}");

    Ok(())
}
