//! Forkable read cursors over a shared [`Bytes`](crate::Bytes) buffer.
//!
//! See [`Bytes::fork`](crate::Bytes::fork) for an introduction.

use alloc::sync::Arc;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::{Buf, Bytes};

/// A read cursor over a [`Bytes`] buffer that belongs to a pair of forked
/// cursors.
///
/// A `BytesCursor` is created together with a sibling by calling
/// [`Bytes::fork()`]. The two cursors share the same underlying storage and
/// the same initial position. Each cursor can be advanced independently
/// through [`Buf`] methods and later [merged](BytesCursor::merge) back
/// together without copying the underlying data.
///
/// # Merge rules
///
/// Two cursors can be merged only when all of the following hold:
///
/// - they are siblings produced by the *same* call to [`Bytes::fork()`]
///   (cursors from different forks are *foreign*);
/// - neither of them is *expired*, i.e. the fork pair has not already been
///   merged;
/// - they still share the same initial position and their remaining ranges
///   are compatible: both remaining ranges must end at the same position, so
///   that one remaining range is a suffix of the other.
///
/// On a successful merge the two cursors become *expired* (they can still be
/// read from, but can no longer take part in a merge) and the merge returns a
/// zero-copy [`Bytes`] view covering the bytes not consumed by either cursor.
/// On a failed merge neither cursor is modified and both remain readable.
#[must_use = "a cursor does nothing unless read from"]
pub struct BytesCursor {
    /// The remaining, not yet consumed view. Advanced through [`Buf`].
    view: Bytes,

    /// The shared initial position of the fork pair (invariant: never moves).
    origin: *const u8,

    /// The shared end position of the original fork range (invariant).
    end: *const u8,

    /// State shared with the sibling cursor.
    state: Arc<ForkState>,
}

struct ForkState {
    /// Set to `true` when the fork pair is merged successfully.
    merged: AtomicBool,
}

impl BytesCursor {
    pub(crate) fn new_pair(view: Bytes) -> (BytesCursor, BytesCursor) {
        let origin = view.as_ptr();
        let end = unsafe { view.as_ptr().add(view.len()) };
        let state = Arc::new(ForkState {
            merged: AtomicBool::new(false),
        });

        let first = BytesCursor {
            view: view.clone(),
            origin,
            end,
            state: state.clone(),
        };
        let second = BytesCursor {
            view,
            origin,
            end,
            state,
        };
        (first, second)
    }

    /// Returns `true` if this cursor's fork pair has already been merged.
    ///
    /// Expired cursors stay readable but reject every further merge attempt.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Buf, Bytes};
    ///
    /// let bytes = Bytes::from_static(b"hello");
    /// let (mut left, mut right) = bytes.fork();
    /// assert!(!left.is_expired());
    ///
    /// left.advance(2);
    /// let _ = left.merge(&mut right);
    /// assert!(left.is_expired());
    /// assert!(right.is_expired());
    /// ```
    pub fn is_expired(&self) -> bool {
        self.state.merged.load(Ordering::Acquire)
    }

    /// Gets a reference to the remaining bytes of this cursor.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Buf, Bytes};
    ///
    /// let bytes = Bytes::from_static(b"hello");
    /// let (mut cursor, _sibling) = bytes.fork();
    /// assert_eq!(&cursor.get_ref()[..], b"hello");
    ///
    /// cursor.advance(2);
    /// assert_eq!(&cursor.get_ref()[..], b"llo");
    /// ```
    pub fn get_ref(&self) -> &Bytes {
        &self.view
    }

    /// Consumes the cursor, returning its remaining [`Bytes`] view.
    ///
    /// This does not affect the sibling cursor or the fork pair's merge
    /// state.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Buf, Bytes};
    ///
    /// let bytes = Bytes::from_static(b"hello");
    /// let (mut cursor, _sibling) = bytes.fork();
    /// cursor.advance(2);
    /// assert_eq!(cursor.into_inner(), Bytes::from_static(b"llo"));
    /// ```
    pub fn into_inner(self) -> Bytes {
        self.view
    }

    /// Attempts to merge this cursor with its forked sibling.
    ///
    /// The merge succeeds only when `self` and `other` are the two cursors
    /// returned by the same [`Bytes::fork()`] call, the pair has not been
    /// merged yet and the remaining ranges are compatible (see the struct
    /// documentation for the exact rules).
    ///
    /// On success both cursors are marked as expired and a zero-copy [`Bytes`]
    /// view of the bytes unread by either cursor is returned. On failure both
    /// cursors are left untouched (nothing is consumed) and remain readable,
    /// and an error describing why the merge was rejected is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Buf, Bytes};
    ///
    /// let bytes = Bytes::from_static(b"hello world");
    /// let (mut left, mut right) = bytes.fork();
    ///
    /// left.advance(6);
    /// right.advance(1);
    ///
    /// let merged = left.try_merge(&mut right).unwrap();
    /// assert_eq!(&merged[..], b"world");
    /// assert!(left.is_expired());
    /// ```
    pub fn try_merge(&mut self, other: &mut BytesCursor) -> Result<Bytes, MergeError> {
        // Cursors must come from the same `fork` call.
        if !Arc::ptr_eq(&self.state, &other.state) {
            return Err(MergeError::Foreign);
        }

        // The fork pair must not have been merged already.
        if self.state.merged.load(Ordering::Acquire) {
            return Err(MergeError::Expired);
        }

        // The initial position is fixed at fork time and must never differ.
        if self.origin != other.origin {
            return Err(MergeError::IncompatibleRange);
        }

        // The two cursors may have advanced independently, so their current
        // (start) positions can differ. The remaining ranges are only
        // compatible when they both end at the original end position and do
        // not start before the shared origin.
        let self_start = self.view.as_ptr();
        let other_start = other.view.as_ptr();
        let self_end = end_ptr(&self.view);
        let other_end = end_ptr(&other.view);

        let within_origin = |start: *const u8| self.origin <= start && start <= self.end;

        if self_end != self.end
            || other_end != self.end
            || !within_origin(self_start)
            || !within_origin(other_start)
        {
            return Err(MergeError::IncompatibleRange);
        }

        // Commit the merge exactly once. The CAS also guards against a
        // concurrent merge of the same pair.
        if self
            .state
            .merged
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(MergeError::Expired);
        }

        // Zero-copy merge: the more advanced cursor leaves the shorter
        // remaining suffix, which is exactly the set of bytes unread by
        // either side, so a cheap `Bytes` clone is all that is needed.
        let merged = if self.view.len() <= other.view.len() {
            self.view.clone()
        } else {
            other.view.clone()
        };

        Ok(merged)
    }

    /// Merges this cursor with its forked sibling.
    ///
    /// This is the infallible counterpart of [`try_merge`](Self::try_merge):
    /// it panics if the cursors do not satisfy the merge rules. On failure
    /// (before the panic) both cursors stay readable and unconsumed.
    ///
    /// # Panics
    ///
    /// Panics if `other` is not the sibling from the same fork call, if the
    /// pair has already been merged, or if the remaining ranges are
    /// incompatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Buf, Bytes};
    ///
    /// let bytes = Bytes::from_static(b"hello world");
    /// let (mut left, mut right) = bytes.fork();
    ///
    /// left.advance(6);
    /// right.advance(6);
    ///
    /// let merged = left.merge(&mut right);
    /// assert_eq!(&merged[..], b"world");
    /// ```
    #[track_caller]
    pub fn merge(&mut self, other: &mut BytesCursor) -> Bytes {
        match self.try_merge(other) {
            Ok(merged) => merged,
            Err(error) => panic!("failed to merge forked cursors: {error}"),
        }
    }
}

fn end_ptr(bytes: &Bytes) -> *const u8 {
    // The `Bytes` view is always contained in its backing storage, so the
    // resulting pointer stays within, or one byte past, the allocation.
    unsafe { bytes.as_ptr().add(bytes.len()) }
}

impl Buf for BytesCursor {
    #[inline]
    fn remaining(&self) -> usize {
        self.view.remaining()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.view.chunk()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        self.view.advance(cnt);
    }

    #[inline]
    fn copy_to_bytes(&mut self, len: usize) -> Bytes {
        self.view.copy_to_bytes(len)
    }
}

impl AsRef<[u8]> for BytesCursor {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.view.as_ref()
    }
}

impl fmt::Debug for BytesCursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesCursor")
            .field("remaining", &self.view.len())
            .field("expired", &self.is_expired())
            .finish()
    }
}

// SAFETY: `Bytes` is `Send + Sync` and `ForkState` only holds an atomic flag,
// so sharing a cursor across threads is as safe as sharing a plain `Bytes`.
unsafe impl Send for BytesCursor {}
unsafe impl Sync for BytesCursor {}

/// An error returned when two forked [`BytesCursor`] values cannot be merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MergeError {
    /// The two cursors do not belong to the same fork pair.
    Foreign,
    /// The fork pair has already been merged, so the cursors are expired.
    Expired,
    /// The cursors do not share the same initial position or their remaining
    /// ranges are not compatible (they do not end at the same position).
    IncompatibleRange,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergeError::Foreign => {
                f.write_str("cannot merge cursors from different forks")
            }
            MergeError::Expired => {
                f.write_str("cannot merge an already merged (expired) fork pair")
            }
            MergeError::IncompatibleRange => f.write_str(
                "cannot merge cursors whose remaining ranges are incompatible",
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MergeError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directly construct an invalid pair to exercise the
    /// `IncompatibleRange` branch without going through the public API.
    #[test]
    fn incompatible_ranges_are_rejected_without_consuming() {
        let bytes = Bytes::from_static(b"abcdef");
        let (mut left, mut right) = BytesCursor::new_pair(bytes);
        // Forge an incompatible invariant: give the right cursor a longer
        // underlying view whose end lies past the fork range's end.
        right.view = Bytes::from_static(b"abcdef_extra");

        assert_eq!(
            left.try_merge(&mut right),
            Err(MergeError::IncompatibleRange)
        );

        // Failure consumes nothing: both views are untouched and still
        // readable, and the pair has not been marked as merged.
        assert_eq!(left.chunk(), b"abcdef");
        assert_eq!(right.chunk(), b"abcdef_extra");
        assert!(!left.is_expired());
        assert!(!right.is_expired());
    }
}
