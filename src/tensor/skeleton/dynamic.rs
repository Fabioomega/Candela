use std::sync::Arc;

use crate::skeleton::Skeleton;
use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
use crate::{Composable, Dimension, Layout, OpError, Tensor};

use super::cache::{BuildFunction, EvictionPolicy, LRUPolicy, SkeletonCache, UnboundedPolicy};
use super::frame::BakedPromise;

/// A cache (group) of skeletons with different shapes
///
/// This is a hashmap abstraction on top of a [`Skeleton`] to enable dynamic shapes.
/// It calls the [`BuildFunction`] every time a group of tensors with never-before-seen
/// layouts arrives, and stores the result in the cache. The cache size and eviction
/// behavior are determined by the chosen policy, which must implement [`EvictionPolicy`].
///
/// The build function must bind its slots in the same order as the layouts it receives,
/// returning a [`Skeleton`] that supports that shape.
///
/// [`Skeleton`]: super::Skeleton
///
/// # Examples
///
/// ```
/// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
/// use candela::{Layout, Tensor};
///
/// // One build rule, reused for whatever shape shows up.
/// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
///     let a = SkeletonSlot::new(inputs[0].clone());
///     (&a * 2.0).into_skeleton(&[a]).unwrap()
/// }));
///
/// // Two different shapes build (and cache) two different skeletons.
/// let out4 = sk.run(&[&Tensor::from_scalar(3.0, &[4])])?;
/// let out8 = sk.run(&[&Tensor::from_scalar(3.0, &[8])])?;
/// assert_eq!(out4.data(), &[6.0; 4]);
/// assert_eq!(out8.data(), &[6.0; 8]);
/// # Ok::<(), candela::OpError>(())
/// ```
pub struct DynamicSkeleton<T, B: Backend = DefaultBackend, P: EvictionPolicy = LRUPolicy> {
    cache: SkeletonCache<Box<[Layout]>, P, T, B>,
    build: BuildFunction<T, B>,
}

impl<T, B: Backend, P: EvictionPolicy> std::fmt::Debug for DynamicSkeleton<T, B, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicSkeleton")
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
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
    /// [`Skeleton`]: super::Skeleton
    ///
    /// # Examples
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, Skeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
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
    #[inline]
    pub fn new(cache_size: usize, build: BuildFunction<T, B>) -> Self {
        Self {
            cache: SkeletonCache::new(cache_size),
            build,
        }
    }

    /// Runs the cached skeleton for the inputs' shapes, building one on a miss.
    ///
    /// Looks up the [`Skeleton`] keyed by the inputs' layouts and runs it,
    /// calling the build function first if no entry exists yet. See
    /// [`Skeleton::run`].
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a + 1.0).into_skeleton(&[a]).unwrap()
    /// }));
    ///
    /// let out = sk.run(&[&Tensor::from_scalar(3.0, &[4])])?;
    /// assert_eq!(out.data(), &[4.0; 4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    #[inline]
    pub fn run(&self, inputs: &[&Tensor<T, B>]) -> Result<Tensor<T, B>, OpError> {
        self.cache.run(inputs, &self.build)
    }

    /// Composes the cached skeleton for the inputs' shapes, building one on a miss.
    ///
    /// Like [`run`], but embeds the skeleton's plan into a [`BakedPromise`]
    /// instead of executing it. See [`Skeleton::compose`].
    ///
    /// [`run`]: DynamicSkeleton::run
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// }));
    ///
    /// // Compose over a lazy promise, not a materialized tensor.
    /// let a = Tensor::from_scalar(1.0, &[4]) + 2.0; // TensorPromise, still unevaluated
    /// let baked = sk.compose(&[&a])?;
    /// assert_eq!(baked.to_promise().materialize().data(), &[6.0; 4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    #[inline]
    pub fn compose<C>(&self, inputs: &[&C]) -> Result<BakedPromise<T, B>, OpError>
    where
        C: Composable<T, B>,
    {
        self.cache.compose(inputs, &self.build)
    }

    /// Removes the entry for `key`
    ///
    /// Returns the skeleton that was stored, or `None` if `key` was not present. The
    /// freed slot is returned to the cache for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// }));
    ///
    /// let a = Tensor::from_scalar(3.0, &[4]);
    /// sk.run(&[&a])?;                     // builds and caches an entry
    /// assert!(sk.remove(&[&a]).is_some());
    /// assert!(!sk.contains_key(&[&a]));   // gone now
    /// # Ok::<(), candela::OpError>(())
    /// ```
    #[inline]
    pub fn remove(&self, key: &[&Tensor<T, B>]) -> Option<Arc<Skeleton<T, B>>> {
        let layouts: Box<[Layout]> = key.iter().map(|&x| x.layout().clone()).collect();

        self.cache.remove(&layouts)
    }

    /// Removes the entry for `key` via layout
    ///
    /// Same as [`Self::remove`] but using layouts instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// }));
    ///
    /// sk.run(&[&Tensor::from_scalar(3.0, &[4])])?;
    /// assert!(sk.remove_by_layout(&[Layout::new(&[4])]).is_some());
    /// # Ok::<(), candela::OpError>(())
    /// ```
    #[inline]
    pub fn remove_by_layout(&self, key: &[Layout]) -> Option<Arc<Skeleton<T, B>>> {
        self.cache.remove(key)
    }

    /// Returns whether `key` currently has an entry in the cache
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// }));
    ///
    /// let a = Tensor::from_scalar(3.0, &[4]);
    /// assert!(!sk.contains_key(&[&a])); // nothing built yet
    /// sk.run(&[&a])?;
    /// assert!(sk.contains_key(&[&a]));  // now cached
    /// # Ok::<(), candela::OpError>(())
    /// ```
    #[inline]
    pub fn contains_key(&self, key: &[&Tensor<T, B>]) -> bool {
        let layouts: Box<[Layout]> = key.iter().map(|&x| x.layout().clone()).collect();

        self.cache.contains_key(&layouts)
    }

    /// Returns whether `key` currently has an entry in the cache
    /// by layout
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::{DynamicSkeleton, SkeletonSlot};
    /// use candela::{Layout, Tensor};
    ///
    /// let sk: DynamicSkeleton<f32> = DynamicSkeleton::new(8, Box::new(|inputs: &[Layout]| {
    ///     let a = SkeletonSlot::new(inputs[0].clone());
    ///     (&a * 2.0).into_skeleton(&[a]).unwrap()
    /// }));
    ///
    /// sk.run(&[&Tensor::from_scalar(3.0, &[4])])?;
    /// assert!(sk.contains_key_by_layout(&[Layout::new(&[4])]));
    /// # Ok::<(), candela::OpError>(())
    /// ```
    #[inline]
    pub fn contains_key_by_layout(&self, key: &[Layout]) -> bool {
        self.cache.contains_key(key)
    }
}

/// A [`DynamicSkeleton`] whose cache never evicts (uses [`UnboundedPolicy`]).
///
/// # Examples
///
/// ```
/// use candela::skeleton::{SkeletonSlot, UnboundedDynamicSkeleton};
/// use candela::{Layout, Tensor};
///
/// // Like DynamicSkeleton, but every skeleton it builds is kept forever.
/// let sk: UnboundedDynamicSkeleton<f32> =
///     UnboundedDynamicSkeleton::new(0, Box::new(|inputs: &[Layout]| {
///         let a = SkeletonSlot::new(inputs[0].clone());
///         (&a * 2.0).into_skeleton(&[a]).unwrap()
///     }));
///
/// let out = sk.run(&[&Tensor::from_scalar(3.0, &[4])])?;
/// assert_eq!(out.data(), &[6.0; 4]);
/// # Ok::<(), candela::OpError>(())
/// ```
pub type UnboundedDynamicSkeleton<T, B = DefaultBackend> = DynamicSkeleton<T, B, UnboundedPolicy>;
