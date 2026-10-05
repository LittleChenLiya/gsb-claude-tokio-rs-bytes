//! Acceptance tests for explicit range view plans.
//!
//! Run locally with:
//!
//! ```text
//! cargo test --test test_range_plan
//! ```

use bytes::{Bytes, BytesMut, RangePlan, RangePlanErrorKind};

fn ptr_in(view: &Bytes, base: *const u8, len: usize) -> bool {
    let p = view.as_ptr() as usize;
    let lo = base as usize;
    let hi = lo + len;
    // Empty views may point one-past-the-end of their range; everything
    // non-empty must lie strictly inside the source storage.
    (lo..=hi).contains(&p) && (view.is_empty() || p + view.len() <= hi)
}

#[test]
fn overlapping_views_share_storage_with_source() {
    let data = Bytes::from((0..64u8).collect::<Vec<_>>());
    let base = data.as_ptr();
    let len = data.len();

    // Two overlapping ranges: neither copied nor merged.
    let plan = RangePlan::from_ranges([(8, 40), (16, 48)]).unwrap();
    let views = data.view_plan(&plan).unwrap();

    assert_eq!(views.len(), 2);
    assert_eq!(&views[0][..], &data[8..40]);
    assert_eq!(&views[1][..], &data[16..48]);

    // Both views point into the exact pointer range of the original storage.
    assert!(ptr_in(&views[0], base, len));
    assert!(ptr_in(&views[1], base, len));
    assert_eq!(views[0].as_ptr(), unsafe { base.add(8) });
    assert_eq!(views[1].as_ptr(), unsafe { base.add(16) });

    // The overlap region is shared, not duplicated: identical addresses.
    let overlap_view0 = views[0].as_ptr() as usize + 8; // offset 16 of data
    let overlap_view1 = views[1].as_ptr() as usize;
    assert_eq!(overlap_view0, overlap_view1);
}

#[test]
fn bad_third_range_fails_whole_batch() {
    let data = Bytes::from(vec![7u8; 32]);

    let plan = RangePlan::from_ranges([(0, 8), (8, 16), (16, 33)]).unwrap();
    let err = data.view_plan(&plan).unwrap_err();
    assert_eq!(err.kind(), RangePlanErrorKind::OutOfBounds);
    assert_eq!(err.index(), 2); // request identity of the bad range
    assert_eq!(err.start(), 16);
    assert_eq!(err.end(), 33);
    assert_eq!(err.len(), 32);

    // A plan that is valid on a longer buffer must still fail wholesale on a
    // shorter one: no partial batch is ever observable.
    let longer = Bytes::from(vec![7u8; 64]);
    assert_eq!(longer.view_plan(&plan).unwrap().len(), 3);
    assert!(data.view_plan(&plan).is_err());
}

#[test]
fn inverted_range_rejected_at_plan_time() {
    let mut plan = RangePlan::new();
    plan.push(0, 4).unwrap();
    let err = plan.push(6, 2).unwrap_err();
    assert_eq!(err.kind(), RangePlanErrorKind::Inverted);
    assert_eq!(err.index(), 1);
    // The failed push did not modify the plan.
    assert_eq!(plan.len(), 1);

    assert!(RangePlan::from_ranges([(0, 1), (3, 3), (5, 4)]).is_err());
}

#[test]
fn views_outlive_dropped_source_handle() {
    let views = {
        let data = Bytes::from(vec![42u8; 16]);
        let plan = RangePlan::from_ranges([(0, 8), (4, 16)]).unwrap();
        data.view_plan(&plan).unwrap()
        // `data` (the original handle) is dropped here.
    };
    assert_eq!(&views[0][..], &[42u8; 8][..]);
    assert_eq!(&views[1][..], &[42u8; 12][..]);
}

#[test]
fn empty_range_keeps_request_identity() {
    let data = Bytes::from_static(b"abcdef");

    // Offsets deliberately out of order, with an empty range in the middle:
    // output must follow request identity, not offset order.
    let plan = RangePlan::from_ranges([(4, 6), (2, 2), (0, 3)]).unwrap();
    let views = data.view_plan(&plan).unwrap();

    assert_eq!(views.len(), 3);
    assert_eq!(&views[0][..], b"ef");
    assert_eq!(&views[1][..], b""); // empty, but still request 1
    assert_eq!(&views[2][..], b"abc");
    assert!(views[1].is_empty());
    assert_eq!(views[1].len(), 0);
}

#[test]
fn four_range_example() {
    let data = Bytes::from_static(b"abcdefghijklmnopqrstuvwxyz");

    let mut plan = RangePlan::new();
    plan.push(0, 4).unwrap(); // "abcd"
    plan.push(4, 4).unwrap(); // empty view, still a request identity
    plan.push(2, 6).unwrap(); // "cdef", overlaps request 0
    plan.push(22, 26).unwrap(); // "wxyz"

    let views = data.view_plan(&plan).unwrap();
    assert_eq!(&views[0][..], b"abcd");
    assert_eq!(&views[1][..], b"");
    assert_eq!(&views[2][..], b"cdef");
    assert_eq!(&views[3][..], b"wxyz");

    // Overlap shares storage: request 2 starts exactly at data offset 2.
    assert_eq!(views[2].as_ptr(), unsafe { data.as_ptr().add(2) });
}

#[test]
fn freeze_views_consumes_bytes_mut() {
    let buf = BytesMut::from(&b"hello world"[..]);
    let plan = RangePlan::from_ranges([(0, 5), (6, 11), (6, 6)]).unwrap();

    // Freeze semantics: the mutable handle is consumed, views share the
    // frozen storage without copying.
    let frozen_ptr = buf.as_ptr();
    let views = buf.freeze_views(&plan).unwrap();
    assert_eq!(&views[0][..], b"hello");
    assert_eq!(&views[1][..], b"world");
    assert_eq!(&views[2][..], b"");
    assert_eq!(views[0].as_ptr(), frozen_ptr);
}

#[test]
fn freeze_views_failure_returns_buffer_and_no_views() {
    let buf = BytesMut::from(&b"short"[..]);
    let plan = RangePlan::from_ranges([(0, 2), (0, 100)]).unwrap();

    let (returned, err) = buf.freeze_views(&plan).unwrap_err();
    assert_eq!(err.kind(), RangePlanErrorKind::OutOfBounds);
    assert_eq!(err.index(), 1);
    // The buffer is handed back untouched and still usable.
    assert_eq!(&returned[..], b"short");
    let views = returned
        .freeze_views(&RangePlan::from_ranges([(0, 5)]).unwrap())
        .unwrap();
    assert_eq!(&views[0][..], b"short");
}

#[test]
fn empty_plan_yields_empty_batch() {
    let data = Bytes::from_static(b"xyz");
    let plan = RangePlan::new();
    assert!(plan.is_empty());
    let views = data.view_plan(&plan).unwrap();
    assert!(views.is_empty());
}
