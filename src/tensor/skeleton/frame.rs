use std::collections::HashMap;
use std::iter::zip;
use std::sync::Arc;

use crate::tensor::allocate::AlignedBuf;
use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
use crate::tensor::executor::{owned_step, run_plan};
use crate::tensor::graph::{NodeKind, TensorGraphBaked, TensorGraphNode, TensorGraphSlot};
use crate::tensor::planner::{
    ALIGNMENT_BYTES, OutputKind, OwnedComputeKind, OwnedPlan, from_borrowed_core_to_owned,
    plan_computation,
};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Composable, Numeric, Operand, Promising};
use crate::{Dimension, Layout, OpError, Tensor, TensorPromise};

/// The input slots of a [`Skeleton`].
///
/// Represents the inputs of a [`Skeleton`] and prevents constructing graphs that
/// cannot be safely materialized. A slot supports all the operations of a
/// [`Tensor`], but produces a [`SkeletonPromise`] instead of a [`TensorPromise`];
/// that promise is then baked - akin to `.materialize()` - through
/// [`into_skeleton`].
///
/// [`into_skeleton`]: SkeletonPromise::into_skeleton
/// [`TensorPromise`]: crate::TensorPromise
///
/// # Examples
///
/// ```
/// use candela::skeleton::SkeletonSlot;
/// use candela::Tensor;
///
/// // A slot stands in for a [4] input; the graph is planned once here...
/// let slot = SkeletonSlot::from_shape(&[4]);
/// let skeleton = (&slot * 2.0 + 1.0).into_skeleton(&[slot])?;
///
/// // ...then run against as many real tensors as you like.
/// let out = skeleton.run(&[&Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4])])?;
/// assert_eq!(out.data(), &[1.0, 3.0, 5.0, 7.0]);
/// # Ok::<(), candela::OpError>(())
/// ```
pub struct SkeletonSlot<T, B: Backend = DefaultBackend> {
    pub(crate) graph: Arc<TensorGraphSlot<T, B>>,
}

impl<T, B: Backend> std::fmt::Debug for SkeletonSlot<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkeletonSlot")
            .field("layout", self.graph.layout())
            .finish()
    }
}

impl<T, B: Backend> SkeletonSlot<T, B> {
    /// Creates a new input slot with the given [`Layout`].
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::{Dimension, Layout};
    ///
    /// let slot: SkeletonSlot<f64> = SkeletonSlot::new(Layout::new(&[2, 3]));
    /// assert_eq!(slot.shape(), &[2, 3]);
    /// ```
    #[inline]
    pub fn new(layout: Layout) -> Self {
        Self {
            graph: Arc::new(TensorGraphSlot::new(layout)),
        }
    }

    /// A deep clone of a Slot
    ///
    /// Equivalent to creating a new slot with the same layout as the old one.
    /// If you just need to reuse the slot use [`clone`] instead.
    ///
    /// [`clone`]: SkeletonSlot::clone
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::Dimension;
    ///
    /// let a: SkeletonSlot<f64> = SkeletonSlot::from_shape(&[4]);
    /// let b = a.deep_clone(); // independent slot, same layout
    /// assert_eq!(a.shape(), b.shape());
    /// ```
    #[inline]
    pub fn deep_clone(&self) -> Self {
        Self::new(self.graph.layout().clone())
    }

    /// Like [`SkeletonSlot::from_shape`], but on an explicit backend `B`. See
    /// [`Tensor`](crate::Tensor) for the `_in` convention.
    #[inline]
    pub fn from_shape_in(shape: &[usize]) -> Self {
        SkeletonSlot::new(Layout::new(shape))
    }
}

impl<T> SkeletonSlot<T, DefaultBackend> {
    /// Creates a new contiguous input slot with the given shape.
    ///
    /// A shorthand for `SkeletonSlot::new(Layout::new(shape))`.
    ///
    /// Uses [`DefaultBackend`]; see [`from_shape_in`](Self::from_shape_in) to
    /// pick a backend explicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::Dimension;
    ///
    /// let slot: SkeletonSlot<f64> = SkeletonSlot::from_shape(&[2, 3]);
    /// assert_eq!(slot.shape(), &[2, 3]);
    /// assert!(slot.is_contiguous());
    /// ```
    #[inline]
    pub fn from_shape(shape: &[usize]) -> Self {
        SkeletonSlot::new(Layout::new(shape))
    }
}

impl<T, B: Backend> Dimension for SkeletonSlot<T, B> {
    fn layout(&self) -> &Layout {
        self.graph.layout()
    }
}

impl<T, B: Backend> Operand<T, B> for SkeletonSlot<T, B> {
    fn to_node(&self) -> NodeKind<T, B> {
        NodeKind::Slot(self.graph.clone())
    }
}

impl<T, B: Backend> Tainting for SkeletonSlot<T, B> {
    type Mark = Tainted;
}

impl<T, B: Backend> Clone for SkeletonSlot<T, B> {
    /// A shallow clone of a Slot
    ///
    /// The copy is equivalent to the slot it was copied from. If you want
    /// a new slot with the same layout, use [`deep_clone`] instead.
    ///
    /// [`deep_clone`]: Self::deep_clone
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// A pre-baked [`Skeleton`] ready to slot into another graph.
///
/// Produced by [`Skeleton::compose`]. It holds the skeleton's plan with its
/// inputs already bound, so it can be used like a regular promise inside
/// operations - it is treated as an opaque node during planning. It can only be
/// materialized through [`to_promise`], because computing it still requires
/// planning.
///
/// [`to_promise`]: BakedPromise::to_promise
pub struct BakedPromise<T, B: Backend> {
    graph: Arc<TensorGraphBaked<T, B>>,
}

impl<T: std::fmt::Debug, B: Backend> std::fmt::Debug for BakedPromise<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.graph, f)
    }
}

impl<T: Clone + PartialEq, B: Backend> BakedPromise<T, B> {
    fn from_node(
        plan: &Arc<OwnedPlan<T, B>>,
        inputs: Box<[NodeKind<T, B>]>,
        inputs_idx: Box<[usize]>,
        layout: &Layout,
    ) -> Self {
        Self {
            graph: Arc::new(TensorGraphBaked::from_node(
                plan, inputs, inputs_idx, layout,
            )),
        }
    }

    /// Creates a fresh [`SkeletonSlot`] matching the shape of this promise's output.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::{Dimension, Tensor};
    ///
    /// let a = Tensor::from_scalar(1.0, &[4]);
    /// let x = SkeletonSlot::from_shape(&[4]);
    /// let baked = (&x * 2.0).into_skeleton(&[x])?.compose(&[&a])?;
    ///
    /// let slot = baked.to_slot(); // fresh slot shaped like the baked output
    /// assert_eq!(slot.shape(), &[4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn to_slot(&self) -> SkeletonSlot<T, B> {
        SkeletonSlot::new(self.layout().clone())
    }
}

impl<T: Numeric, B: Backend> BakedPromise<T, B> {
    /// Wraps the baked computation in a [`TensorPromise`].
    ///
    /// Used mainly when the baked output should act like a regular promise
    /// (e.g. `+=` loops), or simply to materialize it.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::{Layout, Tensor};
    ///
    /// let base_a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
    /// let base_b = Tensor::from_scalar(10.0, &[4]);
    ///
    /// // Two lazy inputs - promises, not materialized tensors.
    /// let a = &base_a + 1.0;
    /// let b = &base_b * 2.0;
    ///
    /// // Compose `x + y` over the two promises, then materialize the result.
    /// let x = SkeletonSlot::new(Layout::new(&[4]));
    /// let y = x.deep_clone();
    /// let baked = (&x + &y).into_skeleton(&[x, y])?.compose(&[&a, &b])?;
    ///
    /// let result = baked.to_promise().materialize();
    /// assert_eq!(result.data(), &[22.0, 23.0, 24.0, 25.0]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn to_promise(&self) -> TensorPromise<T, B> {
        // The promise can always be unwrapped as it's a noop
        unsafe {
            TensorPromise::new(
                crate::tensor::ops::def_op::OpKind::NoOp,
                Box::new([NodeKind::Baked(self.graph.clone())]),
            )
            .unwrap_unchecked()
        }
    }
}

impl<T, B: Backend> Dimension for BakedPromise<T, B> {
    fn layout(&self) -> &Layout {
        self.graph.layout()
    }
}

impl<T, B: Backend> Operand<T, B> for BakedPromise<T, B> {
    fn to_node(&self) -> NodeKind<T, B> {
        NodeKind::Baked(self.graph.clone())
    }
}

impl<T, B: Backend> Tainting for BakedPromise<T, B> {
    type Mark = Clean;
}

impl<T, B: Backend> Composable<T, B> for BakedPromise<T, B> {}

//////////////////////////////////////////////////////////////////////////////////

/// A computation with at least one [`SkeletonSlot`] in its lineage.
///
/// This is the "tainted" sibling of [`TensorPromise`]: every op that touches a
/// slot yields one of these instead of a `TensorPromise`, and it deliberately
/// has no `materialize` - the only way out is [`into_skeleton`], which binds
/// the slots and compiles the plan.
///
/// [`into_skeleton`]: SkeletonPromise::into_skeleton
pub struct SkeletonPromise<T, B: Backend>(TensorPromise<T, B>);

impl<T: std::fmt::Debug, B: Backend> std::fmt::Debug for SkeletonPromise<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl<T, B: Backend> SkeletonPromise<T, B> {
    pub(crate) fn from_promise(promise: TensorPromise<T, B>) -> Self {
        Self(promise)
    }
}

impl<T: ComputeFor<B>, B: Backend> SkeletonPromise<T, B> {
    /// Bakes the recorded computation into a reusable [`Skeleton`].
    ///
    /// `slots` must be the list of slots used during the construction of this
    /// graph. Their order is the order [`Skeleton::run`] and [`Skeleton::compose`]
    /// expect their inputs.
    ///
    /// Planning happens once, during the construction of the skeleton.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::Tensor;
    ///
    /// let slot = SkeletonSlot::from_shape(&[4]);
    /// // Building over the slot yields a SkeletonPromise; into_skeleton compiles it.
    /// let skeleton = (&slot + 10.0).into_skeleton(&[slot])?;
    ///
    /// let out = skeleton.run(&[&Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4])])?;
    /// assert_eq!(out.data(), &[10.0, 11.0, 12.0, 13.0]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`OpError::IncorrectSlotAmount`] if the number of `slots` differs
    /// from the number the computation depends on, or [`OpError::NotSameSlot`] if
    /// a provided slot was never used while building it.
    pub fn into_skeleton(self, slots: &[SkeletonSlot<T, B>]) -> Result<Skeleton<T, B>, OpError> {
        let declared: Vec<(usize, Layout)> = slots
            .iter()
            .map(|s| (s.graph.id, s.layout().clone()))
            .collect();

        Skeleton::from_node(&self.0.graph, declared)
    }
}

impl<T, B: Backend> Dimension for SkeletonPromise<T, B> {
    #[inline]
    fn layout(&self) -> &Layout {
        self.0.layout()
    }
}

impl<T, B: Backend> Operand<T, B> for SkeletonPromise<T, B> {
    fn to_node(&self) -> NodeKind<T, B> {
        self.0.to_node()
    }
}

impl<T, B: Backend> Tainting for SkeletonPromise<T, B> {
    type Mark = Tainted;
}

//////////////////////////////////////////////////////////////////////////////////
// Taint algebra
//
// Every `Operand` carries a `Mark`: `Clean` for a materializable value,
// `Tainted` for anything with a slot in its lineage (`SkeletonSlot`,
// `SkeletonPromise`). An op's output wrapper is the join of its operands'
// marks - `Tainted` is absorbing - so a slot anywhere in an expression forces a
// `SkeletonPromise`, which has no `materialize`.

pub struct Clean;
pub struct Tainted;

/// Taint marker for an operand: `Clean` for a materializable value, `Tainted`
/// when a [`SkeletonSlot`] is in its lineage.
pub trait Tainting {
    type Mark;
}

/// Join of two marks. `Tainted` absorbs `Clean`.
pub trait JoinMark<Rhs> {
    type Out;
}
impl JoinMark<Clean> for Clean {
    type Out = Clean;
}
impl JoinMark<Tainted> for Clean {
    type Out = Tainted;
}
impl JoinMark<Clean> for Tainted {
    type Out = Tainted;
}
impl JoinMark<Tainted> for Tainted {
    type Out = Tainted;
}

/// Maps a mark to the concrete promise wrapper, with the constructor that turns
/// the raw graph result an op produces into that wrapper.
pub trait Wrap<T, B: Backend> {
    type Output;
    fn wrap(promise: TensorPromise<T, B>) -> Self::Output;
}
impl<T, B: Backend> Wrap<T, B> for Clean {
    type Output = TensorPromise<T, B>;
    #[inline]
    fn wrap(promise: TensorPromise<T, B>) -> TensorPromise<T, B> {
        promise
    }
}
impl<T, B: Backend> Wrap<T, B> for Tainted {
    type Output = SkeletonPromise<T, B>;
    #[inline]
    fn wrap(promise: TensorPromise<T, B>) -> SkeletonPromise<T, B> {
        SkeletonPromise::from_promise(promise)
    }
}

/// Output of a unary op on a single operand: wrapped by the operand's own mark.
pub trait UnaryResult<T, B: Backend> {
    type Output;
    fn wrap(promise: TensorPromise<T, B>) -> Self::Output;
}
impl<L, T, B: Backend> UnaryResult<T, B> for L
where
    L: Tainting,
    L::Mark: Wrap<T, B>,
{
    type Output = <L::Mark as Wrap<T, B>>::Output;
    #[inline]
    fn wrap(promise: TensorPromise<T, B>) -> Self::Output {
        <L::Mark as Wrap<T, B>>::wrap(promise)
    }
}

/// Output of a binary op on two operands: wrapped by the join of their marks.
pub trait BinaryResult<Rhs, T, B: Backend> {
    type Output;
    fn wrap(promise: TensorPromise<T, B>) -> Self::Output;
}
impl<L, R, T, B: Backend> BinaryResult<R, T, B> for L
where
    L: Tainting,
    R: Tainting,
    L::Mark: JoinMark<R::Mark>,
    <L::Mark as JoinMark<R::Mark>>::Out: Wrap<T, B>,
{
    type Output = <<L::Mark as JoinMark<R::Mark>>::Out as Wrap<T, B>>::Output;
    #[inline]
    fn wrap(promise: TensorPromise<T, B>) -> Self::Output {
        <<L::Mark as JoinMark<R::Mark>>::Out as Wrap<T, B>>::wrap(promise)
    }
}

//////////////////////////////////////////////////////////////////////////////////

/// A precompiled execution plan, built once and run many times against new inputs.
///
/// # Examples
///
/// ```
/// use candela::Tensor;
/// use std::error::Error;
///
/// // Creates tensors
/// let a = Tensor::from_scalar(0.3, &[4]);
/// let b = Tensor::from_scalar(0.3, &[8]);
///
/// // Creates a slot for a tensor with the same shape as a
/// let slot = a.to_slot();
///
/// // Create a skeleton with that slot
/// let skeleton = (&slot * 2.0 + 1.0).log2().into_skeleton(&[slot]).unwrap();
///
/// // Running the skeleton
/// let output_a = skeleton.run(&[&a]);
///
/// // Running the skeleton for an invalid shape
/// let output_b = skeleton.run(&[&b]);
///
/// // Check the output is ok
/// assert!(output_a.is_ok());
///
/// // Check the output is an error
/// assert!(output_b.is_err());
/// ```
pub struct Skeleton<T, B: Backend = DefaultBackend> {
    plan: Arc<OwnedPlan<T, B>>,
    declared_slots: Vec<(usize, Layout)>,
    layout: Layout,
}

impl<T, B: Backend> std::fmt::Debug for Skeleton<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Skeleton")
            .field("declared_slots", &self.declared_slots)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl<T: Clone + PartialEq + ComputeFor<B>, B: Backend> Skeleton<T, B> {
    pub(crate) fn from_node(
        node: &TensorGraphNode<T, B>,
        declared_slots: Vec<(usize, Layout)>,
    ) -> Result<Self, OpError> {
        let plan = plan_computation(node);

        if plan.external_inputs.len() != declared_slots.len() {
            return Err(OpError::IncorrectSlotAmount(
                plan.external_inputs.len(),
                declared_slots.len(),
            ));
        }

        // Every declared slot must correspond to a slot the plan actually needs.
        // Match them up by id, removing each as it's found so duplicates are handled
        // correctly; a declared slot with no match was never used in the graph.
        let mut external_ids: Vec<usize> = plan.external_inputs.clone();

        for (slot_id, _) in &declared_slots {
            match external_ids.iter().position(|id| id == slot_id) {
                Some(pos) => {
                    external_ids.swap_remove(pos);
                }
                None => return Err(OpError::NotSameSlot(*slot_id)),
            }
        }

        Ok(Self {
            plan: Arc::new(from_borrowed_core_to_owned(plan)),
            declared_slots,
            layout: node.layout().clone(),
        })
    }

    /// Executes the compiled plan against `inputs` and returns the result.
    ///
    /// Runs the stored plan on the provided inputs without re-planning. The
    /// `inputs` must be supplied in the same order they were declared to
    /// [`into_skeleton`].
    ///
    /// [`into_skeleton`]: SkeletonPromise::into_skeleton
    ///
    /// # Errors
    ///
    /// Returns [`OpError::IncorrectSlotAmount`] if `inputs.len()` differs from
    /// the number of declared slots, or [`OpError::NotSameLayoutAtSlot`] if an
    /// input's [`Layout`] does not match the layout its slot was declared with.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::{Layout, Tensor};
    ///
    /// // The same compiled plan, executed against two different inputs.
    /// let slot = SkeletonSlot::new(Layout::new(&[4]));
    /// let skeleton = (&slot * 2.0 + 1.0).into_skeleton(std::slice::from_ref(&slot))?;
    ///
    /// let a = skeleton.run(&[&Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4])])?;
    /// let b = skeleton.run(&[&Tensor::from_scalar(5.0, &[4])])?;
    /// assert_eq!(a.data(), &[1.0, 3.0, 5.0, 7.0]);
    /// assert_eq!(b.data(), &[11.0; 4]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn run(&self, inputs: &[&Tensor<T, B>]) -> Result<Tensor<T, B>, OpError> {
        if inputs.len() != self.declared_slots.len() {
            return Err(OpError::IncorrectSlotAmount(
                self.declared_slots.len(),
                inputs.len(),
            ));
        }

        for ((i, t), (_, layout)) in zip(inputs.iter().enumerate(), self.declared_slots.iter()) {
            if t.layout() != layout {
                return Err(OpError::NotSameLayoutAtSlot(i));
            }
        }

        let external: Vec<(usize, TensorData<T>)> = zip(inputs.iter(), self.declared_slots.iter())
            .map(|(t, (id, _))| (*id, t.graph.compute()))
            .collect();

        let allocated_arena: AlignedBuf<T> = AlignedBuf::new(self.plan.arena_size, ALIGNMENT_BYTES);

        let output = run_plan(
            &mut self.plan.plan.iter().map(owned_step),
            self.plan.root_id,
            external,
            allocated_arena.as_ptr(),
        );

        Ok(Tensor::from_data(output))
    }

    /// Embeds the compiled plan as a node in a larger graph.
    ///
    /// Embeds the [`Skeleton`]'s plan into a promise that must still be planned
    /// and materialized to produce a [`Tensor`]. Unlike [`run`], its inputs may
    /// be any [`Composable`] operand except a slot - [`Tensor`], [`TensorPromise`],
    /// or [`BakedPromise`].
    ///
    /// For all practical purposes, treat the output of this function as a
    /// compressed representation of a [`TensorPromise`].
    ///
    /// [`run`]: Skeleton::run
    /// [`Composable`]: crate::Composable
    /// [`TensorPromise`]: crate::TensorPromise
    ///
    /// # Errors
    ///
    /// Returns [`OpError::IncorrectSlotAmount`] if `inputs.len()` differs from
    /// the number of declared slots, or [`OpError::NotSameLayoutAtSlot`] if an
    /// input's [`Layout`] does not match the layout its slot was declared with.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    /// use candela::{Layout, Tensor};
    ///
    /// let lhs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
    /// let rhs = Tensor::from_scalar(10.0, &[4]);
    ///
    /// // Compile `a + b` over two slots, then splice it into a bigger expression.
    /// let a = SkeletonSlot::new(Layout::new(&[4]));
    /// let b = a.deep_clone();
    /// let sum = (&a + &b).into_skeleton(&[a, b])?;
    ///
    /// let baked = sum.compose(&[&lhs, &rhs])?;
    /// // `baked` slots into a normal promise expression.
    /// let result = (baked * 2.0).materialize();
    /// assert_eq!(result.data(), &[22.0, 24.0, 26.0, 28.0]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn compose<C: Composable<T, B>>(
        &self,
        inputs: &[&C],
    ) -> Result<BakedPromise<T, B>, OpError> {
        if inputs.len() != self.declared_slots.len() {
            return Err(OpError::IncorrectSlotAmount(
                self.declared_slots.len(),
                inputs.len(),
            ));
        }

        for ((i, t), (_, layout)) in zip(inputs.iter().enumerate(), self.declared_slots.iter()) {
            if t.layout() != layout {
                return Err(OpError::NotSameLayoutAtSlot(i));
            }
        }

        let inputs_idx: Box<[usize]> = self.declared_slots.iter().map(|(id, _)| *id).collect();
        let inputs: Vec<NodeKind<T, B>> = inputs.iter().map(|x| x.to_node()).collect();

        Ok(BakedPromise::from_node(
            &self.plan.clone(),
            inputs.into_boxed_slice(),
            inputs_idx,
            &self.layout,
        ))
    }
}

impl<T, B: Backend> Dimension for Skeleton<T, B> {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

//////////////////////////////////////////////////////////////////////////////////
/// A snapshot of the allocations a [`Skeleton`] will perform when run.
///
/// Returned by [`Skeleton::memory_report`]; every field is in bytes and reflects
/// the plan's cache state at the moment the report was taken.
///
/// # Note
///
/// All allocations are reported as Candela sees them; they do not account for
/// caching or memory reuse by the system allocator, so the figures may differ
/// from what actually happens at runtime.
///
/// Candela also uses a handful of bookkeeping allocations that are not
/// counted. For anything but the smallest tensors this is noise next to the
/// buffers above; it only becomes comparable to the work when a step does very
/// little of it.
///
/// # Examples
///
/// ```
/// use candela::skeleton::SkeletonSlot;
///
/// let slot = SkeletonSlot::from_shape(&[4]);
/// let skeleton = (&slot * 2.0 + 1.0).into_skeleton(&[slot])?;
///
/// let report = skeleton.memory_report();
/// // Every field is in bytes; a [4] f64 output is 4 * 8 = 32 bytes.
/// assert_eq!(report.output_memory_usage, 32);
/// assert!(report.peak_memory_usage >= report.output_memory_usage);
/// # Ok::<(), candela::OpError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMetrics {
    /// the size of the allocated (in bytes)
    arena_size: usize,
}

impl<T, B: Backend> Skeleton<T, B> {
    fn memory_report_plan(&self, root_id: usize, plan: &[OwnedComputeKind<T, B>]) -> MemoryMetrics {
        for compute_kind in plan.iter() {
            match compute_kind {
                OwnedComputeKind::Op {
                    node,
                    output,
                    resolved_inputs,
                    dealloc_after,
                } => match output {
                    OutputKind::Allocate(size) => {}
                    OutputKind::Region { offset, len } => {
                        todo!();
                    }
                    OutputKind::InPlaceIdx { idx, .. } => {}
                    OutputKind::Reference(_) => {}
                },
                OwnedComputeKind::CachedOp {
                    cache,
                    output,
                    resolved_inputs,
                    dealloc_after,
                    ..
                } => {
                    if cache.is_cache_filled() {
                        // If the cache is filled the planner assigns no allocations but, in case of data races,
                        // the planner saw an unfilled cache. Then, it emits something that is not an Allocate.
                        // In that case, the executor removes the allocation made for the cache.
                        match output {
                            OutputKind::InPlaceIdx { idx, .. } => {}
                            OutputKind::Region { offset, len } => {}
                            _ => {}
                        }
                    } else {
                        if let OutputKind::Allocate(size) = output {}
                    }
                }
                OwnedComputeKind::Baked {
                    baked,
                    dealloc_after,
                    ..
                } => {
                    let report = self.memory_report_plan(baked.plan.root_id, &baked.plan.plan);
                }
                OwnedComputeKind::Leaf { .. } => {}
            }
        }

        MemoryMetrics { arena_size: 0 }
    }

    /// Reports memory allocations
    ///
    /// Reports the memory that will be allocated during the execution of the [`Skeleton`].
    /// The report is correct *at the moment* this function was called, but changes to
    /// cache state (filled vs empty) after it was run *will* change the metrics.
    ///
    /// For the most accurate results rerun this function every time a cache part of this
    /// [`Skeleton`] is changed (even by itself on the first run).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::skeleton::SkeletonSlot;
    ///
    /// let slot = SkeletonSlot::from_shape(&[8]);
    /// let skeleton = (&slot * 2.0).into_skeleton(&[slot])?;
    ///
    /// let report = skeleton.memory_report();
    /// assert!(report.total_number_of_allocations >= 1);
    /// assert_eq!(report.output_memory_usage, 64); // [8] f64 = 64 bytes
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn memory_report(&self) -> MemoryMetrics {
        self.memory_report_plan(self.plan.root_id, &self.plan.plan)
    }
}
