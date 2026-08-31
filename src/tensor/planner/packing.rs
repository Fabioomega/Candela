use std::cmp::Reverse;

use crate::tensor::planner::ALIGNMENT_BYTES;
use crate::tensor::planner::runtime::Slot;

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub const fn alignment_of<T>() -> usize {
    match size_of::<T>() {
        0 => 1,
        size => ALIGNMENT_BYTES / gcd(ALIGNMENT_BYTES, size),
    }
}

////////////////////////////////////////////////////////////////

pub struct PackedSlot {
    pub id: usize,
    pub start: usize,
    pub end: usize,
    pub offset: usize,
    pub len: usize,
}

// Black magic to get alignment for powers of 2
const fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

fn fit_len(slots: &[PackedSlot], start: usize, end: usize, len: usize, alignment: usize) -> usize {
    let mut last_offset: usize = 0;
    let mut best_offset: Option<usize> = None;
    let mut best_gap: usize = usize::MAX;

    for s in slots {
        if s.start <= end && start <= s.end {
            let aligned_offset = align_up(last_offset, alignment);

            if s.offset > aligned_offset {
                let gap = s.offset - aligned_offset;

                if gap >= len && gap - len < best_gap {
                    best_offset = Some(aligned_offset);
                    best_gap = gap - len;
                }
            }

            last_offset = last_offset.max(s.offset + s.len);
        }
    }

    best_offset.unwrap_or_else(|| align_up(last_offset, alignment))
}

pub struct PackedSlots {
    pub arena_size: usize,
    pub slots: Vec<PackedSlot>,
}

pub fn greedy_offset_pack_slots(mut slots: Vec<Slot>, alignment: usize) -> PackedSlots {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );

    slots.sort_unstable_by_key(|s| (Reverse(s.len), s.start));

    let mut packed_slots: Vec<PackedSlot> = Vec::with_capacity(slots.len());
    let mut last_offset: usize = 0;

    for s in slots.into_iter() {
        // Skips packing if the element must live beyond the arena
        if s.end.is_none() {
            continue;
        }

        let fit = fit_len(&packed_slots, s.start, s.end.unwrap(), s.len, alignment);
        let packed = PackedSlot {
            id: s.id,
            start: s.start,
            end: s.end.unwrap(),
            offset: fit,
            len: s.len,
        };
        last_offset = last_offset.max(fit + s.len);

        let pos = packed_slots.partition_point(|s| s.offset <= fit);
        packed_slots.insert(pos, packed);
    }

    PackedSlots {
        arena_size: last_offset,
        slots: packed_slots,
    }
}

#[cfg(test)]
#[path = "packing_tests.rs"]
mod tests;
