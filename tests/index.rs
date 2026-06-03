mod common;

use candela::{OpError, Tensor, s};

// ── get ───────────────────────────────────────────────────────────────────────

#[test]
fn get_1d() {
    let t = Tensor::from_slice(&[10.0, 20.0, 30.0], &[3]);
    assert_eq!(t.get(&[0]).unwrap(), &10.0);
    assert_eq!(t.get(&[2]).unwrap(), &30.0);
}

#[test]
fn get_2d() {
    // [[0,1,2],[3,4,5]]
    let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    assert_eq!(t.get(&[0, 0]).unwrap(), &0.0);
    assert_eq!(t.get(&[1, 2]).unwrap(), &5.0);
}

#[test]
fn get_wrong_rank() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    assert!(matches!(t.get(&[0, 0]), Err(OpError::NotEnoughAxes(1, 2))));
}

#[test]
fn get_out_of_bounds() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    assert!(matches!(t.get(&[3]), Err(OpError::IndexOutOfBounds)));
}

#[test]
fn get_sliced() {
    // [[0,1,2],[3,4,5],[6,7,8]], slice rows [1..3] - non-contiguous view
    let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 3]);
    let sliced = t.slice(s![1..3, ..]).unwrap().materialize();
    // sliced[0][1] == 4.0, sliced[1][2] == 8.0
    assert_eq!(sliced.get(&[0, 1]).unwrap(), &4.0);
    assert_eq!(sliced.get(&[1, 2]).unwrap(), &8.0);
}

#[test]
fn get_transposed() {
    // [[0,1,2],[3,4,5]].T - [i,j] in transposed space reads [j,i] of original
    let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let tr = t.transpose().materialize();
    // tr is [3,2]: tr[2][1] == original[1][2] == 5.0
    assert_eq!(tr.get(&[2, 1]).unwrap(), &5.0);
}

// ── Index ─────────────────────────────────────────────────────────────────────

#[test]
fn index_1d() {
    let t = Tensor::from_slice(&[7.0, 8.0, 9.0], &[3]);
    assert_eq!(t[&[1][..]], 8.0);
}

#[test]
fn index_2d() {
    // [[1,2],[3,4]]
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(t[&[1, 0][..]], 3.0);
}

// ── item ──────────────────────────────────────────────────────────────────────

#[test]
fn item_scalar() {
    let t = Tensor::from_scalar(42.0_f64, &[1]);
    assert_eq!(t.item(), &42.0);
}

#[test]
fn item_after_materialize() {
    // Ops that reduce to a single element - item reads it correctly
    let t = Tensor::from_scalar(3.0_f64, &[1]);
    let result = (t * 7.0).materialize();
    assert_eq!(result.item(), &21.0);
}
