use crate::buf::UninitSlice;
use crate::{BufMut, TryPutError};

use core::cmp;

/// A `BufMut` adapter which limits the amount of bytes that can be written
/// to an underlying buffer, borrowing the buffer instead of consuming it.
///
/// Unlike [`Limit`](super::Limit), this adapter only borrows the underlying
/// buffer, so the buffer can be used again once the adapter is dropped.
/// Besides the infallible [`BufMut`] interface (which panics on overflow,
/// exactly like `Limit`), the adapter provides
/// [`try_put_slice`](TryLimit::try_put_slice), which rejects a single write
/// that would exceed the remaining limit instead of panicking. A rejected
/// write is rejected as a whole: no bytes are written and the cursor is not
/// partially advanced.
///
/// A `TryLimit` value is created with [`BufMut::try_limit`].
#[derive(Debug)]
pub struct TryLimit<'a, T: BufMut + ?Sized> {
    inner: &'a mut T,
    limit: usize,
}

pub(super) fn new<'a, T: BufMut + ?Sized>(inner: &'a mut T, limit: usize) -> TryLimit<'a, T> {
    TryLimit { inner, limit }
}

impl<'a, T: BufMut + ?Sized> TryLimit<'a, T> {
    /// Consumes this `TryLimit`, returning the underlying buffer.
    pub fn into_inner(self) -> &'a mut T {
        self.inner
    }

    /// Gets a reference to the underlying `BufMut`.
    ///
    /// It is inadvisable to directly write to the underlying `BufMut`.
    pub fn get_ref(&self) -> &T {
        self.inner
    }

    /// Gets a mutable reference to the underlying `BufMut`.
    ///
    /// It is inadvisable to directly write to the underlying `BufMut`.
    pub fn get_mut(&mut self) -> &mut T {
        self.inner
    }

    /// Returns the maximum number of bytes that can be written.
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
        self.limit = lim;
    }

    /// Writes the whole of `src` into `self`, rejecting the write as a whole
    /// if it would exceed the remaining limit.
    ///
    /// Unlike [`BufMut::put_slice`], which panics when there is not enough
    /// remaining capacity, this method returns a [`TryPutError`] and leaves
    /// the buffer untouched: no bytes are written and the cursor is not
    /// partially advanced.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{BufMut, TryPutError};
    ///
    /// let mut buf = Vec::with_capacity(8);
    /// let mut dst = buf.try_limit(3);
    ///
    /// dst.try_put_slice(b"ab").unwrap();
    /// assert_eq!(
    ///     Err(TryPutError { requested: 2, available: 1 }),
    ///     dst.try_put_slice(b"cd"),
    /// );
    /// // The rejected write did not advance the cursor.
    /// dst.try_put_slice(b"c").unwrap();
    /// drop(dst);
    ///
    /// assert_eq!(buf, b"abc");
    /// ```
    pub fn try_put_slice(&mut self, src: &[u8]) -> Result<(), TryPutError> {
        let available = self.remaining_mut();
        if src.len() > available {
            return Err(TryPutError {
                requested: src.len(),
                available,
            });
        }

        self.inner.put_slice(src);
        self.limit -= src.len();
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
