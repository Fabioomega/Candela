use std::iter::zip;
use std::sync::Arc;

use crate::tensor::backend::{Backend, ComputeFor};
use crate::tensor::executor::{owned_step, run_plan};
use crate::tensor::graph::{NodeKind, TensorGraphBaked, TensorGraphNode, TensorGraphSlot};
use crate::tensor::planner::{OwnedCorePlan, core_plan_computation, from_borrowed_core_to_owned};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Composable, Numeric, Promising};
use crate::{Dimension, Layout, OpError, Tensor, TensorPromise};

pub struct SkeletonSlot<T, B: Backend> {
    pub(crate) graph: Arc<TensorGraphSlot<T, B>>,
}

impl<T, B: Backend> SkeletonSlot<T, B> {
    pub fn new(layout: Layout) -> Self {
        Self {
            graph: Arc::new(TensorGraphSlot::new(layout)),
        }
    }
}

impl<T, B: Backend> Dimension for SkeletonSlot<T, B> {
    fn layout(&self) -> &Layout {
        self.graph.layout()
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

impl<T, B: Backend> Composable<T, B> for BakedPromise<T, B> {
    fn to_node(&self) -> NodeKind<T, B> {
        NodeKind::Baked(self.graph.clone())
    }
}

//////////////////////////////////////////////////////////////////////////////////

pub struct SkeletonPromise<T, B: Backend>(TensorPromise<T, B>);

impl<T: ComputeFor<B>, B: Backend> SkeletonPromise<T, B> {
    fn from_promise(promise: TensorPromise<T, B>) -> Self {
        Self(promise)
    }

    pub fn into_skeleton(&self, slots: &[SkeletonSlot<T, B>]) -> Result<Skeleton<T, B>, OpError> {
        let declared: Vec<(usize, Layout)> = slots
            .iter()
            .map(|s| (s.graph.id, s.layout().clone()))
            .collect();

        Skeleton::from_node(&self.0.graph, declared)
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

    pub fn run(&self, inputs: &[Tensor<T, B>]) -> Result<Tensor<T, B>, OpError> {
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
        inputs: &[C],
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
