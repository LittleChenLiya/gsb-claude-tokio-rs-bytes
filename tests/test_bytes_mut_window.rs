//! Tests for [`BytesMut::window`] — the explicit protection window.
//!
//! These tests cover the acceptance criteria:
//! - reading `ab` then writing `cd` does not overwrite the window;
//! - after `commit` the confirmed prefix can be reclaimed;
//! - dropping the window without committing leaves the data intact;
//! - writes that exceed capacity are rejected before any bytes are written;
//! - growth cannot happen while the window borrows the buffer (Rust borrowing
//!   boundary).
//!
//! Run with:
//!
//! ```sh
//! cargo test --test test_bytes_mut_window
//! ```

use bytes::{BufMut, BytesMut};

/// Reading `ab` then writing `cd` must not overwrite the window. The borrowed
/// prefix stays valid and unchanged across the write.
#[test]
fn read_ab_write_cd_does_not_overwrite_window() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let mut win = buf.window();

    // Borrow the parsed prefix.
    let parsed = win.chunk();
    assert_eq!(parsed, b"ab");

    // Append `cd`. This must not invalidate or overwrite `parsed`.
    win.put_slice(b"cd");

    // The borrowed slice is still `ab` and the window now holds `abcd`.
    assert_eq!(parsed, b"ab");
    assert_eq!(win.chunk(), b"abcd");
}

/// The pointer handed out by `chunk` is stable across writes that fit in the
/// existing spare capacity (no reallocation, no compaction).
#[test]
fn chunk_pointer_is_stable_across_writes() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let mut win = buf.window();
    let ptr_before = win.chunk().as_ptr();

    win.put_slice(b"cd");

    let ptr_after = win.chunk().as_ptr();
    assert_eq!(ptr_before, ptr_after, "window pointer must not move");
    assert_eq!(win.chunk(), b"abcd");
}

/// Writing exactly the available spare capacity succeeds and does not panic.
#[test]
fn write_exact_spare_capacity_succeeds() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let mut win = buf.window();
    assert_eq!(win.remaining_mut(), 4);

    win.put_slice(b"cdef");
    assert_eq!(win.chunk(), b"abcdef");
    assert_eq!(win.remaining_mut(), 0);
}

/// A write larger than the spare capacity is rejected (panics) before any
/// bytes are written. The window data is left intact.
#[test]
fn over_capacity_write_is_rejected_before_writing() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let mut win = buf.window();
    assert_eq!(win.remaining_mut(), 4);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        win.put_slice(b"cdefg"); // 5 bytes, only 4 fit
    }));
    assert!(result.is_err(), "over-capacity write must panic");

    // No bytes were written; the window is unchanged.
    assert_eq!(win.chunk(), b"ab");
    assert_eq!(win.remaining_mut(), 4);
}

/// `commit` releases the confirmed prefix and retains the unread suffix.
#[test]
fn commit_releases_prefix_retains_suffix() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"abcd");

    let mut win = buf.window();
    assert_eq!(win.chunk(), b"abcd");

    win.commit(2);

    // The unread suffix (`cd`) is retained.
    assert_eq!(win.chunk(), b"cd");
    assert_eq!(win.remaining(), 2);
}

/// After `commit`, the confirmed prefix space can be reclaimed (compacted) by
/// a subsequent `reserve` once the window is dropped.
#[test]
fn committed_prefix_can_be_reclaimed() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"abcd");

    let mut win = buf.window();
    win.commit(2);
    drop(win);

    // The prefix space (2 bytes before the current start) is reclaimed.
    buf.reserve(4);
    assert_eq!(buf.capacity(), 6);
    assert_eq!(&buf[..], b"cd");
}

/// Dropping the window without committing leaves all data intact.
#[test]
fn drop_without_commit_preserves_data() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    {
        let win = buf.window();
        let parsed = win.chunk();
        assert_eq!(parsed, b"ab");
        // Drop without committing.
    }

    assert_eq!(&buf[..], b"ab");
}

/// The six-byte example: a 6-byte buffer, read `ab`, write `cd`, commit.
#[test]
fn six_byte_example() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let mut win = buf.window();
    let parsed = win.chunk();
    assert_eq!(parsed, b"ab");

    win.put_slice(b"cd");
    assert_eq!(parsed, b"ab");
    assert_eq!(win.chunk(), b"abcd");

    win.commit(2);
    assert_eq!(win.chunk(), b"cd");

    drop(win);
    assert_eq!(&buf[..], b"cd");
}

/// `remaining` / `remaining_mut` introspection.
#[test]
fn introspection() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let win = buf.window();
    assert_eq!(win.remaining(), 2);
    assert_eq!(win.remaining_mut(), 4);
}

/// The window implements [`BufMut`], so the typed `put_*` methods work and
/// share the same no-reallocation guarantee.
#[test]
fn bufmut_put_methods_work() {
    let mut buf = BytesMut::with_capacity(6);
    buf.put_slice(b"ab");

    let mut win = buf.window();
    win.put_u8(b'c');
    win.put_u8(b'd');
    assert_eq!(win.chunk(), b"abcd");
}
