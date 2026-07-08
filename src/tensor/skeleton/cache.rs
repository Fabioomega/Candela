use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
use crate::{Composable, Dimension, Layout, OpError, Tensor};

use super::frame::{BakedPromise, Skeleton};

/// Decides which entry a [`SkeletonCache`] drops when the cache is full.
///
/// The cache calls a hook on each action (insertion, removal, get). The policy
/// keeps whatever bookkeeping it needs and answers [`evict`] when a new entry
/// needs space. Implemented by [`LRUPolicy`] and [`UnboundedPolicy`].
///
/// [`evict`]: EvictionPolicy::evict
///
/// # Examples
///
/// ```
/// // The two built-in policies plug in as a SkeletonCache's third type parameter.
/// use candela::skeleton::{LRUPolicy, SkeletonCache, UnboundedPolicy};
/// use candela::Layout;
///
/// let _lru: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
/// let _unbounded: SkeletonCache<Box<[Layout]>, UnboundedPolicy, f32> = SkeletonCache::new(0);
/// ```
pub trait EvictionPolicy {
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

/// An [`EvictionPolicy`] that never evicts
///
/// The cache grows without bound, keeping every skeleton it has ever built. Use it
/// when the set of input shapes is small and known to be finite.
///
/// # Examples
///
/// ```
/// use candela::skeleton::{SkeletonSlot, UnboundedDynamicSkeleton};
/// use candela::{Layout, Tensor};
///
/// // Selected here through the UnboundedDynamicSkeleton alias.
/// let sk: UnboundedDynamicSkeleton<f32> =
///     UnboundedDynamicSkeleton::new(0, Box::new(|inputs: &[Layout]| {
///         let a = SkeletonSlot::new(inputs[0].clone());
///         (&a * 2.0).into_skeleton(&[a]).unwrap()
///     }));
/// assert_eq!(sk.run(&[&Tensor::from_scalar(3.0, &[4])])?.data(), &[6.0; 4]);
/// # Ok::<(), candela::OpError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct UnboundedPolicy;

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

#[derive(Debug)]
struct Link {
    prev: usize,
    next: usize,
}

/// A least-recently-used [`EvictionPolicy`]
///
/// Tracks access order in an intrusive doubly linked list and, once the cache is
/// full, evicts the entry that has gone longest without a hit.
///
/// # Examples
///
/// ```
/// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
/// use candela::{Layout, Tensor};
///
/// // LRUPolicy is DynamicSkeleton's default; a size-1 cache drops the older shape.
/// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(1, Box::new(|inputs: &[Layout]| {
///     let a = SkeletonSlot::new(inputs[0].clone());
///     (&a * 2.0).into_skeleton(&[a]).unwrap()
/// }));
///
/// let a = Tensor::from_scalar(3.0, &[4]);
/// sk.run(&[&a])?;
/// sk.run(&[&Tensor::from_scalar(3.0, &[8])])?; // evicts the [4] entry
/// assert!(!sk.contains_key(&[&a]));
/// # Ok::<(), candela::OpError>(())
/// ```
#[derive(Debug)]
pub struct LRUPolicy {
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

/// A concurrent store of skeletons keyed by `Key`
///
/// Holds several skeletons at once and picks one by key. Eviction is delegated to
/// the chosen [`EvictionPolicy`]. This is the primitive [`DynamicSkeleton`] is
/// built on; reach for that first unless you need a custom key.
///
/// [`DynamicSkeleton`]: crate::skeleton::DynamicSkeleton
///
/// # Examples
///
/// ```
/// use candela::skeleton::{BuildFunction, LRUPolicy, SkeletonCache, SkeletonSlot};
/// use candela::{Layout, Tensor};
///
/// // Keyed by input layouts, evicting under an LRU policy.
/// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
/// let build: BuildFunction<f32> = Box::new(|inputs: &[Layout]| {
///     let a = SkeletonSlot::new(inputs[0].clone());
///     (&a * 2.0).into_skeleton(&[a]).unwrap()
/// });
///
/// let out = cache.run(&[&Tensor::from_scalar(3.0, &[4])], &build)?;
/// assert_eq!(out.data(), &[6.0; 4]);
/// # Ok::<(), candela::OpError>(())
/// ```
pub struct SkeletonCache<Key: Clone + Hash + Eq, P: EvictionPolicy, T, B: Backend = DefaultBackend>(
    Mutex<SkeletonCacheInner<Key, P, T, B>>,
);

impl<Key: Clone + Hash + Eq, P: EvictionPolicy, T, B: Backend> std::fmt::Debug
    for SkeletonCache<Key, P, T, B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut db_struct = f.debug_struct("SkeletonCache");
        match self.0.try_lock() {
            Ok(inner) => {
                db_struct.field("cached", &inner.cache.map.len());
                db_struct.field("arena_size", &inner.cache.arena.len());
            }
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                let inner = poisoned.get_ref();
                db_struct.field("cached", &inner.cache.map.len());
                db_struct.field("arena_size", &inner.cache.arena.len());
                db_struct.field("state", &format_args!("<poisoned>"));
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                db_struct.field("state", &format_args!("<locked by another thread>"));
            }
        }
        db_struct.finish_non_exhaustive()
    }
}

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

    /// Creates a new cache
    ///
    /// Reserves room for at least `cache_size` entries. The policy decides whether the
    /// cache stays at that size or grows past it.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{LRUPolicy, SkeletonCache};
    /// use candela::Layout;
    ///
    /// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
    /// # let _ = cache;
    /// ```
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

    /// Looks up `key`, building and inserting on a miss
    ///
    /// Returns the cached skeleton if `key` is present. Otherwise `build` is called,
    /// the result is stored under `key`, and a handle to it is returned. `build` runs
    /// at most once, and only on a miss.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{LRUPolicy, SkeletonCache, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
    /// let key: Box<[Layout]> = Box::new([Layout::new(&[4])]);
    ///
    /// // Built on the first call; a second call with the same key reuses it.
    /// let sk = cache.get_or_insert_with(&key, || {
    ///     let a = SkeletonSlot::from_shape(&[4]);
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// });
    /// assert_eq!(sk.run(&[&Tensor::from_scalar(3.0, &[4])])?.data(), &[6.0; 4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
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

    /// Removes the entry for `key`
    ///
    /// Returns the skeleton that was stored, or `None` if `key` was not present. The
    /// freed slot is returned to the cache for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{LRUPolicy, SkeletonCache, SkeletonSlot};
    /// use candela::Layout;
    ///
    /// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
    /// let key: Box<[Layout]> = Box::new([Layout::new(&[4])]);
    /// cache.get_or_insert_with(&key, || {
    ///     let a = SkeletonSlot::from_shape(&[4]);
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// });
    ///
    /// assert!(cache.remove(&key).is_some());
    /// assert!(!cache.contains_key(&key));
    /// ```
    pub fn remove<Q>(&self, key: &Q) -> Option<Arc<Skeleton<T, B>>>
    where
        Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut lock = self.0.lock().unwrap();

        let idx = lock.cache.map.remove(key)?;
        let slot = lock.cache.arena[idx].take();
        lock.free.push(idx);
        lock.policy.on_remove(idx);

        Some(slot.unwrap().sk)
    }

    /// Returns whether `key` currently has an entry in the cache
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{LRUPolicy, SkeletonCache, SkeletonSlot};
    /// use candela::Layout;
    ///
    /// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
    /// let key: Box<[Layout]> = Box::new([Layout::new(&[4])]);
    ///
    /// assert!(!cache.contains_key(&key));
    /// cache.get_or_insert_with(&key, || {
    ///     let a = SkeletonSlot::from_shape(&[4]);
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// });
    /// assert!(cache.contains_key(&key));
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Key: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let lock = self.0.lock().unwrap();

        lock.cache.map.contains_key(key)
    }
}

/// Builds a [`Skeleton`] for a given set of input layouts
///
/// Called on a cache miss with the layouts of the current inputs. It must create its
/// slots from those layouts and bind them in the same order, so the resulting skeleton
/// accepts exactly that shape.
///
/// # Examples
///
/// ```
/// use candela::skeleton::{BuildFunction, SkeletonSlot};
/// use candela::Layout;
///
/// // Doubles whatever single input it is handed, whatever its shape.
/// let build: BuildFunction<f32> = Box::new(|inputs: &[Layout]| {
///     let a = SkeletonSlot::new(inputs[0].clone());
///     (&a * 2.0).into_skeleton(&[a]).unwrap()
/// });
/// # let _ = build;
/// ```
pub type BuildFunction<T, B = DefaultBackend> =
    Box<dyn Fn(&[Layout]) -> Skeleton<T, B> + Send + Sync>;

impl<P, T, B> SkeletonCache<Box<[Layout]>, P, T, B>
where
    P: EvictionPolicy,
    T: Clone + PartialEq + ComputeFor<B>,
    B: Backend,
{
    /// Runs the cached skeleton for the inputs' shapes, building one on a miss.
    ///
    /// Keys the cache by the inputs' layouts; on a miss `on_miss` builds the
    /// [`Skeleton`], which is then cached and run. See [`Skeleton::run`].
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{BuildFunction, LRUPolicy, SkeletonCache, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
    /// let build: BuildFunction<f32> = Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a + 1.0).into_skeleton(&[a]).unwrap()
    /// });
    ///
    /// let out = cache.run(&[&Tensor::from_scalar(3.0, &[4])], &build)?;
    /// assert_eq!(out.data(), &[4.0; 4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn run(
        &self,
        inputs: &[&Tensor<T, B>],
        on_miss: &BuildFunction<T, B>,
    ) -> Result<Tensor<T, B>, OpError> {
        // TODO: this clones every input layout on each call, even on a cache hit where
        // nothing owned is needed. A raw_entry-based lookup could build the owned key only
        // on a miss and skip the allocation entirely on hits.
        let input_layouts: Box<[Layout]> = inputs.iter().map(|&t| t.layout().clone()).collect();

        let sk: Arc<Skeleton<T, B>> =
            self.get_or_insert_with(&input_layouts, || on_miss(&input_layouts));
        sk.run(inputs)
    }

    /// Composes the cached skeleton for the inputs' shapes, building one on a miss.
    ///
    /// Like [`run`], but embeds the skeleton's plan into a [`BakedPromise`]
    /// instead of executing it. See [`Skeleton::compose`].
    ///
    /// [`run`]: SkeletonCache::run
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{BuildFunction, LRUPolicy, SkeletonCache, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let cache: SkeletonCache<Box<[Layout]>, LRUPolicy, f32> = SkeletonCache::new(4);
    /// let build: BuildFunction<f32> = Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// });
    ///
    /// // Compose over a lazy promise and fold the result into a larger graph.
    /// let a = Tensor::from_scalar(1.0, &[4]) + 2.0;
    /// let baked = cache.compose(&[&a], &build)?;
    /// assert_eq!(baked.to_promise().materialize().data(), &[6.0; 4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
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
