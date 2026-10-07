#![warn(rust_2018_idioms)]

use bytes::{buf::MergeError, Buf, Bytes};

// A small, deterministic in-memory fixture: a six byte buffer.
const DATA: &[u8; 6] = b"abcdef";

#[test]
fn fork_starts_at_same_position_and_shares_storage() {
    let bytes = Bytes::from_static(DATA);
    let (left, right) = bytes.fork();

    assert_eq!(left.remaining(), 6);
    assert_eq!(right.remaining(), 6);
    assert_eq!(left.chunk(), DATA);
    assert_eq!(right.chunk(), DATA);
    assert!(!left.is_expired());
    assert!(!right.is_expired());

    // The cursors point into the exact same storage as the original `Bytes`.
    assert_eq!(left.as_ref().as_ptr(), bytes.as_ptr());
    assert_eq!(right.as_ref().as_ptr(), bytes.as_ptr());
}

#[test]
fn legal_merge_after_advancing_different_lengths() {
    let bytes = Bytes::from_static(DATA);
    let (mut left, mut right) = bytes.fork();

    // Advance the two siblings by different lengths.
    left.advance(2);
    right.advance(4);

    assert_eq!(left.remaining(), 4);
    assert_eq!(right.remaining(), 2);

    let merged = left.try_merge(&mut right).expect("merge must succeed");

    // The merge yields the bytes unread by either cursor: the suffix starting
    // at the furthest advanced position.
    assert_eq!(merged, b"ef"[..]);
    assert_eq!(merged.len(), 2);

    // No copy: the merged view points into the original shared storage.
    assert_eq!(
        merged.as_ptr() as usize,
        bytes.as_ptr() as usize + 4,
    );

    // Both cursors are now expired, but they stay readable where they were.
    assert!(left.is_expired());
    assert!(right.is_expired());
    assert_eq!(left.chunk(), b"cdef");
    assert_eq!(right.chunk(), b"ef");
}

#[test]
fn merge_is_symmetrical() {
    let bytes = Bytes::from_static(DATA);
    let (mut left, mut right) = bytes.fork();

    left.advance(4);
    right.advance(2);

    // Passing the further advanced cursor as `self` gives the same result.
    let merged = right.try_merge(&mut left).expect("merge must succeed");
    assert_eq!(merged, b"ef"[..]);
    assert!(left.is_expired());
    assert!(right.is_expired());
}

#[test]
fn equal_length_advance_merges_to_full_remaining() {
    let bytes = Bytes::from_static(DATA);
    let (mut left, mut right) = bytes.fork();

    left.advance(3);
    right.advance(3);

    let merged = left.merge(&mut right);
    assert_eq!(merged, b"def"[..]);
}

#[test]
fn foreign_cursors_are_rejected_and_keep_readable_state() {
    let bytes = Bytes::from_static(DATA);

    // Two independent forks over the same storage are foreign to each other.
    let (mut left_a, _right_a) = bytes.fork();
    let (_left_b, mut right_b) = bytes.fork();

    left_a.advance(1);
    right_b.advance(2);

    let before_left = (left_a.remaining(), left_a.chunk().as_ptr());
    let before_right = (right_b.remaining(), right_b.chunk().as_ptr());

    let error = left_a
        .try_merge(&mut right_b)
        .expect_err("foreign forks must not merge");
    assert_eq!(error, MergeError::Foreign);

    // Failure consumes nothing: positions, remaining bytes and pointers are
    // unchanged and both cursors can still be read.
    assert_eq!((left_a.remaining(), left_a.chunk().as_ptr()), before_left);
    assert_eq!(
        (right_b.remaining(), right_b.chunk().as_ptr()),
        before_right,
    );
    assert_eq!(left_a.chunk(), b"bcdef");
    assert_eq!(right_b.chunk(), b"cdef");
    assert!(!left_a.is_expired());
    assert!(!right_b.is_expired());
}

#[test]
fn expired_cursors_are_rejected_on_a_second_merge() {
    let bytes = Bytes::from_static(DATA);
    let (mut left, mut right) = bytes.fork();

    left.advance(2);
    right.advance(4);

    let first = left.try_merge(&mut right).expect("first merge succeeds");
    assert_eq!(first, b"ef"[..]);

    // A second attempt with the same pair is rejected as expired regardless
    // of merge direction.
    assert_eq!(left.try_merge(&mut right), Err(MergeError::Expired));
    assert_eq!(right.try_merge(&mut left), Err(MergeError::Expired));

    // Expiration does not consume the cursors: they are still positioned where
    // they were and remain fully readable.
    assert_eq!(left.chunk(), b"cdef");
    assert_eq!(right.chunk(), b"ef");
    left.advance(1);
    assert_eq!(left.chunk(), b"def");
}

#[test]
fn a_failed_merge_with_a_foreign_cursor_does_not_expire_either_fork() {
    let bytes = Bytes::from_static(DATA);
    let (mut a1, mut a2) = bytes.fork();
    let (mut b1, mut b2) = bytes.fork();

    a1.advance(1);
    b1.advance(1);

    // Cross-pair merge fails.
    assert_eq!(a1.try_merge(&mut b1), Err(MergeError::Foreign));

    // Each original pair can still merge its own siblings.
    let merged_a = a1.try_merge(&mut a2).unwrap();
    assert_eq!(&merged_a[..], b"bcdef");
    let merged_b = b1.try_merge(&mut b2).unwrap();
    assert_eq!(&merged_b[..], b"bcdef");
}

#[test]
fn merge_does_not_copy_heap_storage() {
    // Heap-allocated (non-static) storage: merging must not duplicate it.
    let bytes = Bytes::from(vec![0u8, 1, 2, 3, 4, 5]);
    let base_ptr = bytes.as_ptr();

    let (mut left, mut right) = bytes.fork();
    left.advance(2);
    right.advance(4);

    let merged = left.merge(&mut right);
    assert_eq!(&merged[..], &[4u8, 5][..]);
    assert_eq!(merged.as_ptr() as usize, base_ptr as usize + 4);
}

#[test]
fn fully_consumed_cursors_merge_to_empty_view() {
    let bytes = Bytes::from_static(DATA);
    let (mut left, mut right) = bytes.fork();

    left.advance(6);
    right.advance(6);

    let merged = left.merge(&mut right);
    assert!(merged.is_empty());
}

#[test]
fn empty_buffer_fork_merges() {
    let bytes = Bytes::new();
    let (mut left, mut right) = bytes.fork();
    assert_eq!(left.remaining(), 0);

    let merged = left.try_merge(&mut right).expect("empty fork merges");
    assert!(merged.is_empty());
    assert!(left.is_expired());
}
