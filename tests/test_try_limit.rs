#![warn(rust_2018_idioms)]

use bytes::{buf::TryLimit, BufMut, BytesMut, TryPutError};

#[test]
fn try_limit_two_writes_fill_exact_capacity() {
    let mut buf = BytesMut::with_capacity(64);
    {
        let mut dst = buf.try_limit(8);
        assert_eq!(8, dst.remaining_mut());
        dst.try_put_slice(b"abcd").unwrap();
        dst.try_put_slice(b"efgh").unwrap();
        assert_eq!(0, dst.remaining_mut());
    }
    // Both small writes landed; nothing more, nothing less.
    assert_eq!(&buf[..], b"abcdefgh");
    assert_eq!(8, buf.len());
}

#[test]
fn try_limit_over_limit_write_is_rejected_without_partial_advance() {
    let mut buf = BytesMut::with_capacity(64);
    {
        let mut dst = buf.try_limit(4);
        dst.try_put_slice(b"ab").unwrap();

        let err = dst.try_put_slice(b"cdef").unwrap_err();
        assert_eq!(
            err,
            TryPutError {
                requested: 4,
                available: 2,
            }
        );

        // The rejected write did not advance the cursor at all.
        assert_eq!(2, dst.remaining_mut());
        assert_eq!(2, TryLimit::limit(&dst));

        // A write that fits the remaining limit still succeeds.
        dst.try_put_slice(b"cd").unwrap();
        assert_eq!(0, dst.remaining_mut());
    }
    // Only the accepted writes are visible; no partial bytes of "cdef".
    assert_eq!(&buf[..], b"abcd");
    assert_eq!(4, buf.len());

    // The adapter borrowed the buffer: after it is dropped, the original
    // buffer is usable again.
    buf.put_slice(b"ef");
    assert_eq!(&buf[..], b"abcdef");
    assert_eq!(6, buf.len());
}

#[test]
fn try_limit_get_ref_and_get_mut() {
    let mut buf = BytesMut::from(&b"hello"[..]);
    let mut dst = buf.try_limit(10);
    assert_eq!(&dst.get_ref()[..], b"hello");
    assert_eq!(dst.get_mut().len(), 5);
}

#[test]
fn try_limit_set_limit() {
    let mut buf = BytesMut::with_capacity(64);
    let mut dst = buf.try_limit(10);
    assert_eq!(10, TryLimit::limit(&dst));
    dst.set_limit(5);
    assert_eq!(5, TryLimit::limit(&dst));
    assert_eq!(5, dst.remaining_mut());
}

#[test]
fn try_limit_chunk_mut() {
    let mut buf = BytesMut::with_capacity(64);
    let mut dst = buf.try_limit(10);
    assert!(dst.chunk_mut().len() <= 10);
}

#[test]
fn try_limit_advance_mut() {
    let mut buf = BytesMut::with_capacity(64);
    let mut dst = buf.try_limit(10);
    unsafe {
        dst.advance_mut(5);
    }
    assert_eq!(5, dst.remaining_mut());
    assert_eq!(5, TryLimit::limit(&dst));
}

#[test]
#[should_panic = "cnt <= self.limit"]
fn try_limit_advance_mut_panic() {
    let mut buf = BytesMut::with_capacity(64);
    let mut dst = buf.try_limit(10);
    unsafe {
        dst.advance_mut(50);
    }
}

#[test]
fn try_put_error_display_and_io_conversion() {
    let err = TryPutError {
        requested: 4,
        available: 2,
    };
    assert_eq!(
        err.to_string(),
        "Not enough space remaining in buffer to write value (requested 4 but only 2 available)"
    );
    let io_err: std::io::Error = err.into();
    assert_eq!(std::io::ErrorKind::WriteZero, io_err.kind());
}
