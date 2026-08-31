use super::*;
use proptest::prelude::*;

type SlotSpec = (usize, usize, usize, Option<usize>);

fn build(specs: &[SlotSpec]) -> Vec<Slot> {
    specs
        .iter()
        .map(|&(id, len, start, end)| Slot {
            id,
            len,
            start,
            end,
        })
        .collect()
}

const fn live(id: usize, len: usize, start: usize, end: usize) -> SlotSpec {
    (id, len, start, Some(end))
}

const fn eternal(id: usize, len: usize, start: usize) -> SlotSpec {
    (id, len, start, None)
}

fn lifetimes_overlap(a: &PackedSlot, b: &PackedSlot) -> bool {
    a.start <= b.end && b.start <= a.end
}

fn ranges_overlap(a: &PackedSlot, b: &PackedSlot) -> bool {
    if a.len == 0 || b.len == 0 {
        return false;
    }

    a.offset < b.offset + b.len && b.offset < a.offset + a.len
}

fn check_invariants(specs: &[SlotSpec], alignment: usize) -> PackedSlots {
    let packed = greedy_offset_pack_slots(build(specs), alignment);

    for (i, a) in packed.slots.iter().enumerate() {
        for b in packed.slots.iter().skip(i + 1) {
            assert!(
                !(lifetimes_overlap(a, b) && ranges_overlap(a, b)),
                "slots {} [{}..={}] @ {}..{} and {} [{}..={}] @ {}..{} are alive together \
                 and share memory",
                a.id,
                a.start,
                a.end,
                a.offset,
                a.offset + a.len,
                b.id,
                b.start,
                b.end,
                b.offset,
                b.offset + b.len,
            );
        }
    }

    for s in &packed.slots {
        assert!(
            s.offset + s.len <= packed.arena_size,
            "slot {} runs to {} past the arena end {}",
            s.id,
            s.offset + s.len,
            packed.arena_size,
        );

        assert_eq!(
            s.offset % alignment,
            0,
            "slot {} sits at {}, not a multiple of {}",
            s.id,
            s.offset,
            alignment,
        );
    }

    let high_water = packed
        .slots
        .iter()
        .map(|s| s.offset + s.len)
        .max()
        .unwrap_or(0);
    assert_eq!(packed.arena_size, high_water);

    let mut expected: Vec<usize> = specs
        .iter()
        .filter(|s| s.3.is_some())
        .map(|s| s.0)
        .collect();
    let mut got: Vec<usize> = packed.slots.iter().map(|s| s.id).collect();
    expected.sort_unstable();
    got.sort_unstable();
    assert_eq!(expected, got);

    assert!(
        packed.slots.windows(2).all(|w| w[0].offset <= w[1].offset),
        "output is not sorted by offset",
    );

    packed
}

fn at(packed: &PackedSlots, id: usize) -> &PackedSlot {
    packed
        .slots
        .iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("slot {id} was not placed"))
}

#[test]
fn alignment_of_common_types() {
    assert_eq!(alignment_of::<u8>(), 128);
    assert_eq!(alignment_of::<u16>(), 64);
    assert_eq!(alignment_of::<f32>(), 32);
    assert_eq!(alignment_of::<f64>(), 16);
}

#[test]
fn alignment_of_zst_is_one() {
    assert_eq!(alignment_of::<()>(), 1);
}

#[test]
fn alignment_of_is_a_power_of_two_block_of_bytes() {
    macro_rules! check {
        ($($t:ty),*) => {$({
            let align = alignment_of::<$t>();
            assert!(
                align.is_power_of_two(),
                "alignment_of::<{}>() = {} is not a power of two",
                stringify!($t),
                align,
            );
            assert_eq!(
                (align * size_of::<$t>()) % ALIGNMENT_BYTES,
                0,
                "alignment_of::<{}>() = {} elements is not a whole number of {}-byte blocks",
                stringify!($t),
                align,
                ALIGNMENT_BYTES,
            );
        })*};
    }

    check!(
        u8, u16, u32, u64, u128, f32, f64, [u8; 3], [u8; 12], [u8; 96], [f64; 5]
    );
}

#[test]
fn empty_input_needs_no_arena() {
    let packed = check_invariants(&[], 32);
    assert_eq!(packed.arena_size, 0);
    assert!(packed.slots.is_empty());
}

#[test]
fn a_single_slot_sits_at_the_base() {
    let packed = check_invariants(&[live(0, 10, 0, 5)], 1);
    assert_eq!(at(&packed, 0).offset, 0);
    assert_eq!(packed.arena_size, 10);
}

#[test]
fn eternal_slots_are_left_out_of_the_arena() {
    let packed = check_invariants(
        &[eternal(0, 1024, 0), eternal(1, 512, 1), eternal(2, 8, 2)],
        32,
    );

    assert!(packed.slots.is_empty());
    assert_eq!(packed.arena_size, 0);
}

#[test]
fn eternal_slots_do_not_reserve_space_around_live_ones() {
    let packed = check_invariants(
        &[live(0, 8, 0, 1), eternal(1, 4096, 0), live(2, 8, 2, 3)],
        1,
    );

    assert_eq!(at(&packed, 0).offset, 0);
    assert_eq!(at(&packed, 2).offset, 0);
    assert_eq!(packed.arena_size, 8);
}

#[test]
fn zero_length_slots_take_no_space() {
    let packed = check_invariants(&[live(0, 0, 0, 5), live(1, 0, 0, 5), live(2, 8, 0, 5)], 1);
    assert_eq!(packed.arena_size, 8);
}

#[test]
fn disjoint_lifetimes_all_share_the_base() {
    let packed = check_invariants(&[live(0, 8, 0, 1), live(1, 8, 2, 3), live(2, 8, 4, 5)], 1);

    for id in 0..3 {
        assert_eq!(
            at(&packed, id).offset,
            0,
            "slot {id} did not reuse the base"
        );
    }
    assert_eq!(packed.arena_size, 8);
}

#[test]
fn simultaneous_slots_stack() {
    let packed = check_invariants(&[live(0, 8, 0, 9), live(1, 4, 0, 9), live(2, 2, 0, 9)], 1);

    assert_eq!(at(&packed, 0).offset, 0);
    assert_eq!(at(&packed, 1).offset, 8);
    assert_eq!(at(&packed, 2).offset, 12);
    assert_eq!(packed.arena_size, 14);
}

#[test]
fn touching_lifetimes_do_not_share() {
    let packed = check_invariants(&[live(0, 8, 0, 5), live(1, 8, 5, 9)], 1);

    assert_ne!(
        at(&packed, 0).offset,
        at(&packed, 1).offset,
        "a slot dying at step 5 shared memory with one born at step 5",
    );
    assert_eq!(packed.arena_size, 16);
}

#[test]
fn dead_regions_are_reused_by_later_slots() {
    let packed = check_invariants(
        &[
            live(0, 1024, 0, 1),
            live(1, 1, 1, 2),
            live(2, 1024, 3, 4),
            live(3, 1, 4, 5),
        ],
        1,
    );

    assert_eq!(at(&packed, 0).offset, at(&packed, 2).offset);
    assert_eq!(at(&packed, 1).offset, at(&packed, 3).offset);
    assert_eq!(packed.arena_size, 1025);
}

#[test]
fn stacked_slots_are_padded_up_to_the_alignment() {
    let packed = check_invariants(&[live(0, 8, 0, 9), live(1, 8, 0, 9)], 32);

    assert_eq!(at(&packed, 0).offset, 0);
    assert_eq!(at(&packed, 1).offset, 32);
    assert_eq!(packed.arena_size, 40);
}

#[test]
fn reuse_still_starts_at_the_base_under_alignment() {
    let packed = check_invariants(&[live(0, 8, 0, 1), live(1, 8, 2, 3)], 32);

    assert_eq!(at(&packed, 0).offset, 0);
    assert_eq!(at(&packed, 1).offset, 0);
}

// Occupied over one window: [0,10), [14,20), [30,40).
// Gaps: 4 elements at 10, 10 elements at 20.
fn gapped_skyline() -> Vec<PackedSlot> {
    vec![
        PackedSlot {
            id: 0,
            start: 0,
            end: 9,
            offset: 0,
            len: 10,
        },
        PackedSlot {
            id: 1,
            start: 0,
            end: 9,
            offset: 14,
            len: 6,
        },
        PackedSlot {
            id: 2,
            start: 0,
            end: 9,
            offset: 30,
            len: 10,
        },
    ]
}

#[test]
fn fit_len_prefers_the_tighter_gap() {
    assert_eq!(fit_len(&gapped_skyline(), 0, 9, 3, 1), 10);
}

#[test]
fn fit_len_takes_a_gap_it_exactly_fills() {
    assert_eq!(fit_len(&gapped_skyline(), 0, 9, 4, 1), 10);
}

#[test]
fn fit_len_skips_a_gap_one_element_too_small() {
    assert_eq!(fit_len(&gapped_skyline(), 0, 9, 5, 1), 20);
}

#[test]
fn fit_len_appends_when_no_gap_is_big_enough() {
    assert_eq!(fit_len(&gapped_skyline(), 0, 9, 11, 1), 40);
}

#[test]
fn fit_len_ignores_slots_that_are_not_alive_together() {
    assert_eq!(fit_len(&gapped_skyline(), 20, 30, 11, 1), 0);
}

#[test]
fn fit_len_aligns_the_gap_it_picks() {
    assert_eq!(fit_len(&gapped_skyline(), 0, 9, 3, 4) % 4, 0);
}

prop_compose! {
    fn arb_specs(max_slots: usize)(
        raw in prop::collection::vec(
            (0usize..64, 0usize..24, 0usize..24, 0u8..8),
            1..=max_slots,
        )
    ) -> Vec<SlotSpec> {
        raw.iter()
            .enumerate()
            .map(|(i, &(len, a, b, eternal_roll))| {
                let (start, end) = (a.min(b), a.max(b));
                let end = if eternal_roll == 0 { None } else { Some(end) };
                (i, len, start, end)
            })
            .collect()
    }
}

proptest! {
    #[test]
    fn packing_never_overlaps(specs in arb_specs(24), shift in 0u32..8) {
        check_invariants(&specs, 1usize << shift);
    }

    #[test]
    fn packing_never_exceeds_the_unpacked_total(specs in arb_specs(24), shift in 0u32..8) {
        let alignment = 1usize << shift;
        let packed = greedy_offset_pack_slots(build(&specs), alignment);

        let unpacked: usize = specs
            .iter()
            .filter(|s| s.3.is_some())
            .map(|s| align_up(s.1, alignment))
            .sum();

        prop_assert!(
            packed.arena_size <= unpacked,
            "packed to {} but {} would fit every slot separately",
            packed.arena_size,
            unpacked,
        );
    }

    #[test]
    fn fully_sequential_slots_need_only_the_largest(
        lens in prop::collection::vec(1usize..512, 1..12),
        shift in 0u32..8,
    ) {
        let specs: Vec<SlotSpec> = lens
            .iter()
            .enumerate()
            .map(|(i, &len)| live(i, len, i * 2, i * 2 + 1))
            .collect();

        let packed = check_invariants(&specs, 1usize << shift);

        prop_assert_eq!(packed.arena_size, *lens.iter().max().unwrap());
    }
}
