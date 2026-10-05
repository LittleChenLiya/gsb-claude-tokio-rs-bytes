//! Tests for [`RangeViews`] — an explicit plan for zero-copy views into one
//! [`Bytes`].
//!
//! Run locally with:
//!
//! ```sh
//! cargo test --test test_range_views
//! ```

use bytes::{Bytes, BytesMut, RangeError, RangeViews};

/// Two overlapping slices share the pointer interval of the original data and
/// are not copied or merged.
#[test]
fn overlapping_views_share_pointer_interval() {
    let data: Vec<u8> = (0..26).collect();
    let base = data.as_ptr();
    let bytes = Bytes::from(data);

    let views = RangeViews::from_ref(&bytes)
        .add(0..10)
        .add(5..15)
        .build()
        .unwrap();

    assert_eq!(views.len(), 2);

    // Every view's pointer lies within the original allocation's interval.
    for view in &views {
        let vp = view.as_ptr();
        assert!(vp >= base, "view pointer {:?} precedes base {:?}", vp, base);
        assert!(
            vp.wrapping_add(view.len()) <= base.wrapping_add(bytes.len()),
            "view overruns the original allocation",
        );
    }

    // The two views overlap each other's address ranges.
    let (a, b) = (&views[0], &views[1]);
    assert!(a.as_ptr() < b.as_ptr().wrapping_add(b.len()));
    assert!(b.as_ptr() < a.as_ptr().wrapping_add(a.len()));

    // They are distinct handles (different offsets), not a merged copy.
    assert_ne!(a.as_ptr(), b.as_ptr());
    assert_eq!(a.len(), 10);
    assert_eq!(b.len(), 10);
}

/// A wrong third range fails the whole batch: no views are returned, and the
/// error identifies the offending request by its request identity.
#[test]
fn invalid_range_fails_entire_batch() {
    let bytes = Bytes::from(&b"abcdef"[..]); // len = 6

    let err = RangeViews::from_ref(&bytes)
        .add(0..2)
        .add(2..4)
        .add(4..10) // out of bounds
        .build()
        .unwrap_err();

    assert_eq!(
        err,
        RangeError {
            request: 2,
            range: 4..10,
            len: 6,
        }
    );
}

/// A reversed range (`start > end`) is rejected as a whole-batch failure.
#[test]
fn reversed_range_fails_batch() {
    let bytes = Bytes::from(&b"abcdef"[..]);

    let err = RangeViews::from_ref(&bytes)
        .add(5..2) // start > end
        .build()
        .unwrap_err();

    assert_eq!(err.request, 0);
    assert_eq!(err.range, 5..2);
    assert_eq!(err.len, 6);
}

/// Dropping the original handle (and the plan) does not affect views already
/// produced — they stay readable because each holds its own reference count.
#[test]
fn views_outlive_original_handle() {
    let views = {
        let data: Vec<u8> = (0..26).collect();
        let bytes = Bytes::from(data);
        RangeViews::new(bytes)
            .add(0..10)
            .add(5..15)
            .add(20..26)
            .build()
            .unwrap()
        // `bytes` and the plan are dropped here.
    };

    assert_eq!(&views[0][..], &(0..10).collect::<Vec<u8>>()[..]);
    assert_eq!(&views[1][..], &(5..15).collect::<Vec<u8>>()[..]);
    assert_eq!(&views[2][..], &(20..26).collect::<Vec<u8>>()[..]);
}

/// Empty ranges are legal and still occupy their request identity in the
/// output.
#[test]
fn empty_ranges_keep_request_identity() {
    let bytes = Bytes::from(&b"abcdef"[..]);

    let views = RangeViews::from_ref(&bytes)
        .add(2..2) // request 0, empty in the middle
        .add(0..3) // request 1, "abc"
        .add(6..6) // request 2, empty at the end
        .add(3..6) // request 3, "def"
        .build()
        .unwrap();

    assert_eq!(views.len(), 4);
    assert!(views[0].is_empty());
    assert_eq!(&views[1][..], b"abc");
    assert!(views[2].is_empty());
    assert_eq!(&views[3][..], b"def");
}

/// Output is in request order, not sorted by offset (the four-range example).
#[test]
fn output_is_request_order_not_offset_order() {
    let bytes = Bytes::from(&b"abcdefghij"[..]);

    let views = RangeViews::from_ref(&bytes)
        .add(7..10)
        .add(0..3)
        .add(4..4)
        .add(3..7)
        .build()
        .unwrap();

    assert_eq!(views.len(), 4);
    assert_eq!(&views[0][..], b"hij");
    assert_eq!(&views[1][..], b"abc");
    assert_eq!(&views[2][..], b"");
    assert_eq!(&views[3][..], b"defg");
}

/// All boundary forms are accepted at the edge of the buffer.
#[test]
fn boundary_forms_are_legal() {
    let bytes = Bytes::from(&b"abcdef"[..]); // len = 6

    let views = RangeViews::from_ref(&bytes)
        .add(0..0) // empty at start
        .add(6..6) // empty at end
        .add(0..=5) // inclusive full range
        .add(..) // unbounded full range
        .add(3..) // unbounded start
        .add(..3) // unbounded end
        .build()
        .unwrap();

    assert_eq!(views.len(), 6);
    assert!(views[0].is_empty());
    assert!(views[1].is_empty());
    assert_eq!(&views[2][..], b"abcdef");
    assert_eq!(&views[3][..], b"abcdef");
    assert_eq!(&views[4][..], b"def");
    assert_eq!(&views[5][..], b"abc");
}

/// The owned constructor consumes the buffer; the borrowed one keeps it usable.
#[test]
fn owned_and_borrowed_constructors() {
    // Owned: the plan manages the buffer.
    let owned = Bytes::from(&b"abc"[..]);
    let views = RangeViews::new(owned).add(0..3).build().unwrap();
    assert_eq!(&views[0][..], b"abc");

    // Borrowed: the buffer is still usable after building.
    let borrowed = Bytes::from(&b"xyz"[..]);
    let mut plan = RangeViews::from_ref(&borrowed);
    let views = plan.add(0..3).build().unwrap();
    assert_eq!(&views[0][..], b"xyz");
    assert_eq!(&borrowed[..], b"xyz");
}

/// A `BytesMut` must be frozen before it can feed a plan; the plan never
/// borrows a mutable handle, so freezing semantics are preserved.
#[test]
fn bytes_mut_must_be_frozen() {
    let mut buf = BytesMut::from(&b"hello"[..]);
    buf.extend_from_slice(b" world");
    let bytes = buf.freeze();

    let views = RangeViews::new(bytes)
        .add(0..5)
        .add(6..11)
        .build()
        .unwrap();

    assert_eq!(&views[0][..], b"hello");
    assert_eq!(&views[1][..], b"world");
}

/// A plan can be built more than once; each build yields independent views.
#[test]
fn plan_is_rebuildable() {
    let bytes = Bytes::from(&b"abcdef"[..]);
    let mut plan = RangeViews::from_ref(&bytes);
    plan.add(0..3);
    plan.add(3..6);

    let first = plan.build().unwrap();
    let second = plan.build().unwrap();

    assert_eq!(first, second);
    assert_eq!(&first[0][..], b"abc");
    assert_eq!(&second[1][..], b"def");
}

/// An empty plan builds to an empty vector.
#[test]
fn empty_plan_builds_to_empty_vec() {
    let bytes = Bytes::from(&b"abc"[..]);
    let views = RangeViews::from_ref(&bytes).build().unwrap();
    assert!(views.is_empty());
}

/// The error type implements `Display` and `std::error::Error`.
#[test]
fn error_is_display_and_error() {
    let bytes = Bytes::from(&b"abc"[..]);
    let err = RangeViews::from_ref(&bytes).add(0..10).build().unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("request 0"), "message was: {msg}");
    assert!(msg.contains("0..10"), "message was: {msg}");
    assert!(msg.contains("length 3"), "message was: {msg}");

    let _: &dyn std::error::Error = &err;
}
