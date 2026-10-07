#![warn(rust_2018_idioms)]

use bytes::{buf::TryLimit, BufMut, TryPutError};

#[test]
fn two_writes_fill_exact_capacity() {
    let mut buf = Vec::with_capacity(8);
    {
        let mut dst = buf.try_limit(4);
        dst.try_put_slice(b"ab").unwrap();
        dst.try_put_slice(b"cd").unwrap();
        assert_eq!(0, dst.remaining_mut());
        assert_eq!(0, dst.chunk_mut().len());
    }
    assert_eq!(&buf[..], b"abcd");
    assert_eq!(4, buf.len());
}

#[test]
fn over_limit_write_is_rejected_without_partial_advance() {
    let mut buf = Vec::with_capacity(8);
    {
        let mut dst = buf.try_limit(3);
        dst.try_put_slice(b"ab").unwrap();

        // A single write exceeding the remaining limit is rejected as a whole.
        assert_eq!(
            Err(TryPutError {
                requested: 2,
                available: 1
            }),
            dst.try_put_slice(b"cd")
        );
        // Nothing was written and the cursor was not partially advanced.
        assert_eq!(1, dst.remaining_mut());

        // A write that fits the remaining limit still succeeds.
        dst.try_put_slice(b"c").unwrap();
        assert_eq!(0, dst.remaining_mut());
    }
    assert_eq!(&buf[..], b"abc");
    assert_eq!(3, buf.len());
}

#[test]
fn buffer_is_usable_after_adapter_is_dropped() {
    let mut buf = Vec::with_capacity(8);
    {
        let mut dst = buf.try_limit(2);
        dst.try_put_slice(b"ab").unwrap();
    }
    // The adapter only borrowed `buf`; after drop the original buffer keeps
    // working and sees the bytes written through the adapter.
    buf.put_slice(b"cd");
    assert_eq!(&buf[..], b"abcd");
    assert_eq!(4, buf.len());
}

#[test]
fn try_limit_limit_and_set_limit() {
    let mut buf = Vec::with_capacity(128);
    let mut dst = buf.try_limit(10);
    assert_eq!(10, TryLimit::limit(&dst));
    assert_eq!(10, dst.remaining_mut());
    dst.set_limit(5);
    assert_eq!(5, TryLimit::limit(&dst));
    assert_eq!(5, dst.remaining_mut());
}

#[test]
fn try_limit_remaining_capped_by_inner() {
    let mut buf = &mut [0u8; 10][..];
    let mut dst = buf.try_limit(100);
    assert_eq!(10, dst.remaining_mut());
    assert_eq!(10, dst.chunk_mut().len());
    assert_eq!(&[0u8; 10], &dst.get_ref()[..]);
    assert_eq!(&mut [0u8; 10], &dst.get_mut()[..]);
}

#[test]
fn try_limit_into_inner() {
    let mut buf = Vec::with_capacity(8);
    {
        let mut dst = buf.try_limit(4);
        dst.try_put_slice(b"ab").unwrap();
        let inner = dst.into_inner();
        inner.put_slice(b"cd");
    }
    assert_eq!(&buf[..], b"abcd");
}

#[test]
fn try_limit_buf_mut_interface_within_limit() {
    let mut buf = Vec::with_capacity(8);
    {
        let mut dst = buf.try_limit(4);
        // The infallible BufMut interface keeps working within the limit.
        dst.put_u16(0x0102);
        dst.put_slice(b"cd");
        assert_eq!(0, dst.remaining_mut());
    }
    assert_eq!(&buf[..], b"\x01\x02cd");
}

#[test]
#[should_panic(expected = "cnt <= self.limit")]
fn try_limit_advance_mut_keeps_bounds() {
    let mut buf = Vec::with_capacity(16);
    let mut dst = buf.try_limit(4);
    // The safety bounds of advance_mut are not loosened by the adapter.
    unsafe {
        dst.advance_mut(5);
    }
}

#[test]
#[should_panic(expected = "advance out of bounds")]
fn try_limit_put_slice_panics_on_overflow() {
    let mut buf = Vec::with_capacity(16);
    let mut dst = buf.try_limit(2);
    // The infallible BufMut interface keeps its panicking behavior.
    dst.put_slice(b"abc");
}
