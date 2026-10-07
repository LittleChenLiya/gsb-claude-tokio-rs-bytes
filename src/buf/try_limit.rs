use crate::buf::UninitSlice;
use crate::{BufMut, TryPutError};

use core::cmp;

/// A `BufMut` adapter which limits the amount of bytes that can be written
/// to an underlying buffer and reports an error instead of partially writing
/// when a single write would exceed the limit.
///
/// Unlike [`Limit`](crate::buf::Limit), which silently truncates writes, the
/// fallible [`try_put_slice`](TryLimit::try_put_slice) method rejects a write
/// that does not fit within the remaining limit as a whole: no bytes are
/// written and the cursor is not advanced.
///
/// The adapter borrows the underlying buffer, so once it is dropped the
/// original buffer can be used (and written to) again.
///
/// This struct is generally created by calling [`try_limit`](BufMut::try_limit)
/// on a `BufMut` value.
#[derive(Debug)]
pub struct TryLimit<'a, T: ?Sized> {
    inner: &'a mut T,
    limit: usize,
}

pub(super) fn new<T: ?Sized>(inner: &mut T, limit: usize) -> TryLimit<'_, T> {
    TryLimit { inner, limit }
}

impl<'a, T: ?Sized> TryLimit<'a, T> {
    /// Gets a reference to the underlying `BufMut`.
    ///
    /// It is inadvisable to directly write to the underlying `BufMut`.
    pub fn get_ref(&self) -> &T {
        &*self.inner
    }

    /// Gets a mutable reference to the underlying `BufMut`.
    ///
    /// It is inadvisable to directly write to the underlying `BufMut`.
    pub fn get_mut(&mut self) -> &mut T {
        &mut *self.inner
    }

    /// Returns the maximum number of bytes that can still be written.
    ///
    /// # Note
    ///
    /// If the inner `BufMut` has fewer bytes than indicated by this method
    /// then that is the actual number of available bytes.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Sets the maximum number of bytes that can be written.
    ///
    /// # Note
    ///
    /// If the inner `BufMut` has fewer bytes than `lim` then that is the
    /// actual number of available bytes.
    pub fn set_limit(&mut self, lim: usize) {
        self.limit = lim
    }
}

impl<T: BufMut + ?Sized> TryLimit<'_, T> {
    /// Attempts to write the entire contents of `src` into the underlying
    /// buffer.
    ///
    /// If `src` fits within the remaining limit, it is written and the
    /// current position is advanced by `src.len()`. Otherwise the write is
    /// rejected as a whole: no bytes are written, the cursor is not
    /// advanced, and a [`TryPutError`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{BytesMut, BufMut};
    ///
    /// let mut buf = BytesMut::with_capacity(64);
    /// {
    ///     let mut dst = buf.try_limit(4);
    ///     dst.try_put_slice(b"ab").unwrap();
    ///     assert!(dst.try_put_slice(b"cde").is_err());
    ///     // The rejected write did not advance the cursor.
    ///     dst.try_put_slice(b"cd").unwrap();
    /// }
    /// assert_eq!(&buf[..], b"abcd");
    /// ```
    pub fn try_put_slice(&mut self, src: &[u8]) -> Result<(), TryPutError> {
        let available = self.remaining_mut();
        if src.len() > available {
            return Err(TryPutError {
                requested: src.len(),
                available,
            });
        }
        self.put_slice(src);
        Ok(())
    }
}

unsafe impl<T: BufMut + ?Sized> BufMut for TryLimit<'_, T> {
    fn remaining_mut(&self) -> usize {
        cmp::min(self.inner.remaining_mut(), self.limit)
    }

    fn chunk_mut(&mut self) -> &mut UninitSlice {
        let bytes = self.inner.chunk_mut();
        let end = cmp::min(bytes.len(), self.limit);
        &mut bytes[..end]
    }

    unsafe fn advance_mut(&mut self, cnt: usize) {
        assert!(cnt <= self.limit);
        self.inner.advance_mut(cnt);
        self.limit -= cnt;
    }
}
