use std::collections::HashMap;
use std::iter::zip;
use std::sync::Arc;

use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
use crate::tensor::executor::{owned_step, run_plan};
use crate::tensor::graph::{NodeKind, TensorGraphBaked, TensorGraphNode, TensorGraphSlot};
use crate::tensor::planner::{
    OutputKind, OwnedComputeKind, OwnedCorePlan, core_plan_computation, from_borrowed_core_to_owned,
};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Composable, Numeric, Operand, Promising};
use crate::{Dimension, Layout, OpError, Tensor, TensorPromise};

pub struct SkeletonSlot<T, B: Backend = DefaultBackend> {
    pub(crate) graph: Arc<TensorGraphSlot<T, B>>,
}

impl<T, B: Backend> SkeletonSlot<T, B> {
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
    /// [`clone`]: Tensor::clone
    #[inline]
    pub fn clone_deep(&self) -> Self {
        Self::new(self.graph.layout().clone())
    }
}

impl<T> SkeletonSlot<T, DefaultBackend> {
    #[inline]
    pub fn from_shape(shape: &[usize]) -> Self {
        SkeletonSlot::new(Layout::from_shape(shape, 0))
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
    /// a new slot with the same layout, use [`clone_deep`] instead.
    ///
    /// [`clone_deep`]: Tensor::clone_deep
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

pub struct BakedPromise<T, B: Backend> {
    graph: Arc<TensorGraphBaked<T, B>>,
}

impl<T: Clone + PartialEq, B: Backend> BakedPromise<T, B> {
    fn from_node(
        plan: &Arc<OwnedCorePlan<T, B>>,
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

    pub fn as_slot(&self) -> SkeletonSlot<T, B> {
        SkeletonSlot::new(self.layout().clone())
    }
}

impl<T: Numeric, B: Backend> BakedPromise<T, B> {
    pub fn as_promise(&self) -> TensorPromise<T, B> {
        // The promise can always be unwrapped as it's a noop
        unsafe {
            TensorPromise::new(
                super::ops::def_op::OpKind::NoOp,
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

impl<T, B: Backend> SkeletonPromise<T, B> {
    pub(crate) fn from_promise(promise: TensorPromise<T, B>) -> Self {
        Self(promise)
    }
}

impl<T: ComputeFor<B>, B: Backend> SkeletonPromise<T, B> {
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
pub struct Skeleton<T, B: Backend> {
    plan: Arc<OwnedCorePlan<T, B>>,
    declared_slots: Vec<(usize, Layout)>,
    layout: Layout,
}

impl<T: Clone + PartialEq + ComputeFor<B>, B: Backend> Skeleton<T, B> {
    pub(crate) fn from_node(
        node: &TensorGraphNode<T, B>,
        declared_slots: Vec<(usize, Layout)>,
    ) -> Result<Self, OpError> {
        let plan = core_plan_computation(node);

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

        let output = run_plan(
            &mut self.plan.plan.iter().map(owned_step),
            self.plan.root_id,
            external,
        );

        Ok(Tensor::from_data(output))
    }

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
#[derive(Debug)]
pub struct MemoryMetrics {
    // peak memory usage in bytes
    pub peak_memory_usage: usize,
    // number of allocations performed
    pub total_number_of_allocations: usize,
    // the sizes, in bytes, of all allocated buffers
    pub allocated_buffers_size: Vec<usize>,
    // total memory allocated in bytes
    pub total_memory_allocated: usize,
    // the size, in bytes, of the output node
    pub output_memory_usage: usize,
}

impl<T, B: Backend> Skeleton<T, B> {
    fn memory_report_plan(
        &self,
        root_id: usize,
        plan: &Vec<OwnedComputeKind<T, B>>,
    ) -> MemoryMetrics {
        let mut allocated_slots: HashMap<usize, usize> = HashMap::new();
        let mut allocated_buffers_size: Vec<usize> = Vec::new();
        let mut total_memory_allocated: usize = 0;
        let mut total_number_of_allocations: usize = 0;
        let mut current_memory_usage: usize = 0;
        let mut peak_memory_usage: usize = 0;
        let mut output_memory_usage: usize = 0;

        for compute_kind in plan.iter() {
            match compute_kind {
                OwnedComputeKind::Op {
                    node,
                    output,
                    resolved_inputs,
                    dealloc_after,
                } => match output {
                    OutputKind::Allocate(size) => {
                        let mem: usize = *size * size_of::<T>();
                        allocated_slots.insert(node.id, mem);
                        allocated_buffers_size.push(mem);
                        total_memory_allocated += mem;
                        total_number_of_allocations += 1;
                        current_memory_usage += mem;
                        peak_memory_usage = peak_memory_usage.max(current_memory_usage);

                        for id in dealloc_after.iter() {
                            if let Some(mem) = allocated_slots.remove(id) {
                                current_memory_usage -= mem;
                            }
                        }
                    }
                    OutputKind::Buffer(id) => {
                        if let Some(mem) = allocated_slots.remove(id) {
                            allocated_slots.insert(node.id, mem);
                        }
                    }
                    OutputKind::InPlaceIdx(idx) => {
                        let id = resolved_inputs[*idx];

                        if let Some(mem) = allocated_slots.remove(&id) {
                            allocated_slots.insert(node.id, mem);
                        }
                    }
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
                            OutputKind::Buffer(id) => {
                                if let Some(mem) = allocated_slots.remove(id) {
                                    current_memory_usage -= mem;
                                }
                            }
                            OutputKind::InPlaceIdx(idx) => {
                                if let Some(mem) = allocated_slots.remove(&resolved_inputs[*idx]) {
                                    current_memory_usage -= mem;
                                }
                            }
                            _ => {}
                        }

                        for id in dealloc_after.iter() {
                            if let Some(mem) = allocated_slots.remove(id) {
                                current_memory_usage -= mem;
                            }
                        }
                    } else {
                        if let OutputKind::Allocate(size) = output {
                            let mem = *size * size_of::<T>();
                            allocated_slots.insert(cache.get_node().id, mem);
                            allocated_buffers_size.push(mem);
                            total_memory_allocated += mem;
                            total_number_of_allocations += 1;
                            current_memory_usage += mem;
                            peak_memory_usage = peak_memory_usage.max(current_memory_usage);

                            for id in dealloc_after.iter() {
                                if let Some(mem) = allocated_slots.remove(id) {
                                    current_memory_usage -= mem;
                                }
                            }
                        }
                    }
                }
                OwnedComputeKind::Baked {
                    baked,
                    dealloc_after,
                    ..
                } => {
                    let report = self.memory_report_plan(baked.plan.root_id, &baked.plan.plan);

                    allocated_slots.insert(baked.id, report.output_memory_usage);
                    allocated_buffers_size.extend(report.allocated_buffers_size.iter());
                    total_memory_allocated += report.total_memory_allocated;
                    total_number_of_allocations += report.total_number_of_allocations;
                    peak_memory_usage =
                        peak_memory_usage.max(current_memory_usage + report.peak_memory_usage);
                    current_memory_usage += report.output_memory_usage;

                    for id in dealloc_after.iter() {
                        if let Some(mem) = allocated_slots.remove(id) {
                            current_memory_usage -= mem;
                        }
                    }
                }
                OwnedComputeKind::Leaf { .. } => {}
            }
        }

        if let Some(size) = allocated_slots.remove(&root_id) {
            output_memory_usage = size;
        }

        MemoryMetrics {
            peak_memory_usage,
            total_number_of_allocations,
            allocated_buffers_size,
            total_memory_allocated,
            output_memory_usage,
        }
    }

    /// Reports memory allocations
    ///
    /// Reports the memory that will be allocated during the execution of the [`Skeleton`].
    /// The report is correct *at the moment* this function was called, but changes to
    /// cache state (filled vs empty) after it was run *will* change the metrics.
    ///
    /// For the most accurate results rerun this function every time a cache part of this
    /// [`Skeleton`] is changed (even by itself on the first run).
    pub fn memory_report(&self) -> MemoryMetrics {
        self.memory_report_plan(self.plan.root_id, &self.plan.plan)
    }
}
