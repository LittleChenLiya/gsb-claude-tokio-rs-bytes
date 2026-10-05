//! An explicit plan for producing multiple zero-copy views into one [`Bytes`].
//!
//! [`RangeViews`] collects a set of byte-range requests that all refer to the
//! same shared buffer, then materializes every view in a single step. Each
//! view is an independent [`Bytes`] handle that shares the buffer's underlying
//! storage; the memory is never copied and overlapping ranges are never merged.
//!
//! # Request identity and output order
//!
//! Views are returned in **request order** — the order in which ranges were
//! added with [`add`](RangeViews::add) — *not* sorted by offset. The position
//! of a view in the returned [`Vec`] is its request identity: view `i`
//! corresponds to request `i`. This matters when requests arrive out of offset
//! order, as in the [four-range example](#four-range-example) below.
//!
//! # Atomic boundary validation
//!
//! All range boundaries are validated against the buffer length *before* any
//! view is produced. If any range is invalid (a reversed range, or one whose
//! end is past the end of the buffer), [`build`](RangeViews::build) returns a
//! [`RangeError`] identifying the offending request and **no views are
//! returned**. Empty ranges are always legal as long as they lie within the
//! buffer (for example `len..len` and `0..0`).
//!
//! # Zero-copy sharing and lifetime
//!
//! Every view is produced by [`Bytes::slice`], which is `O(1)` and only bumps
//! the storage's reference count. Two overlapping views therefore point into
//! the same pointer interval of the original allocation; they do not copy or
//! merge bytes. Because each view holds its own reference count, dropping the
//! original handle (or the plan, or the borrowed buffer) does not affect views
//! that have already been produced — they stay readable until they are dropped.
//!
//! # Owned vs borrowed
//!
//! A plan can either own its buffer ([`RangeViews::new`] takes an owned
//! [`Bytes`]) or borrow it ([`RangeViews::from_ref`] takes `&Bytes`). In both
//! cases the views produced are independent, owned [`Bytes`] handles; the
//! owned/borrowed distinction only affects how the plan holds the buffer while
//! requests are being collected.
//!
//! Only a *frozen* [`Bytes`] is accepted. A [`BytesMut`] cannot be passed to
//! a plan directly — call [`BytesMut::freeze`] first. This guarantees that
//! producing views can never bypass the freeze guarantee: views are immutable
//! shared handles, and no mutable handle can alias the same memory through the
//! plan.
//!
//! # Four-range example
//!
//! ```
//! use bytes::{Bytes, RangeViews};
//!
//! let bytes = Bytes::from(&b"abcdefghij"[..]);
//!
//! let views = RangeViews::from_ref(&bytes)
//!     .add(7..10) // request 0 -> "hij"
//!     .add(0..3)  // request 1 -> "abc"
//!     .add(4..4)  // request 2 -> "" (empty, still has identity)
//!     .add(3..7)  // request 3 -> "defg"
//!     .build()
//!     .unwrap();
//!
//! // Output is in request order, NOT sorted by offset.
//! assert_eq!(views.len(), 4);
//! assert_eq!(&views[0][..], b"hij");
//! assert_eq!(&views[1][..], b"abc");
//! assert_eq!(&views[2][..], b"");
//! assert_eq!(&views[3][..], b"defg");
//! ```
//!
//! # Running the tests locally
//!
//! ```sh
//! cargo test --test test_range_views
//! ```

use core::fmt;
use core::ops::{Bound, Range, RangeBounds};

use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::Bytes;

/// A plan for materializing multiple zero-copy views into a single [`Bytes`].
///
/// Requests are registered with [`add`](RangeViews::add) in request order.
/// [`build`](RangeViews::build) then validates every boundary first and, only
/// if all of them are legal, returns one [`Bytes`] view per request in request
/// order. See the [module-level documentation](self) for an overview.
///
/// # Examples
///
/// ```
/// use bytes::RangeViews;
///
/// let bytes = bytes::Bytes::from(&b"hello world"[..]);
/// let views = RangeViews::from_ref(&bytes)
///     .add(0..5)
///     .add(6..11)
///     .build()
///     .unwrap();
///
/// assert_eq!(&views[0][..], b"hello");
/// assert_eq!(&views[1][..], b"world");
/// ```
#[derive(Debug, Clone)]
pub struct RangeViews<'a> {
    bytes: Cow<'a, Bytes>,
    requests: Vec<Range<usize>>,
}

impl RangeViews<'static> {
    /// Creates a plan that owns `bytes`.
    ///
    /// The plan holds the (frozen) buffer for its lifetime. Use this when you
    /// have an owned [`Bytes`] and want the plan to manage it. The produced
    /// views are independent handles; they do not borrow from the plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Bytes, RangeViews};
    ///
    /// let bytes = Bytes::from(&b"hello"[..]);
    /// let views = RangeViews::new(bytes)
    ///     .add(0..5)
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(&views[0][..], b"hello");
    /// ```
    pub fn new(bytes: Bytes) -> Self {
        Self {
            bytes: Cow::Owned(bytes),
            requests: Vec::new(),
        }
    }
}

impl<'a> RangeViews<'a> {
    /// Creates a plan that borrows `bytes`.
    ///
    /// The plan holds an immutable reference to the (frozen) buffer. Use this
    /// when you want to keep using the buffer while or after building views.
    /// The produced views are independent owned handles; dropping the buffer
    /// (or the plan) does not affect views already produced.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::{Bytes, RangeViews};
    ///
    /// let bytes = Bytes::from(&b"hello"[..]);
    /// let views = RangeViews::from_ref(&bytes)
    ///     .add(0..5)
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(&views[0][..], b"hello");
    ///
    /// // The borrowed buffer is still usable.
    /// assert_eq!(&bytes[..], b"hello");
    /// ```
    pub fn from_ref(bytes: &'a Bytes) -> Self {
        Self {
            bytes: Cow::Borrowed(bytes),
            requests: Vec::new(),
        }
    }

    /// Adds a range request to the plan.
    ///
    /// Requests are recorded in request order; the request's identity is its
    /// position in this order. The range is *not* validated against the buffer
    /// until [`build`](RangeViews::build) is called, so an invalid range does
    /// not prevent other requests from being recorded — instead the whole batch
    /// fails at build time.
    ///
    /// Empty ranges are legal (for example `4..4`).
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::RangeViews;
    ///
    /// let bytes = bytes::Bytes::from(&b"abcdef"[..]);
    /// let mut plan = RangeViews::from_ref(&bytes);
    /// plan.add(0..2).add(2..4).add(4..6);
    /// assert_eq!(plan.len(), 3);
    /// ```
    pub fn add(&mut self, range: impl RangeBounds<usize>) -> &mut Self {
        let len = self.bytes.len();
        self.requests.push(resolve(range, len));
        self
    }

    /// Returns the number of range requests in the plan.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns `true` if the plan contains no range requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Validates every range request and materializes the views.
    ///
    /// All boundaries are checked first. If any range is invalid (`start > end`
    /// or `end > len`), this returns a [`RangeError`] identifying the first
    /// offending request and no views are produced. Empty ranges are legal as
    /// long as they lie within the buffer.
    ///
    /// On success, views are returned in request order (the order in which they
    /// were added), not sorted by offset. Each view is an independent [`Bytes`]
    /// handle sharing the buffer's storage; overlapping ranges are not copied or
    /// merged.
    ///
    /// Because `build` takes `&self`, the plan can be built more than once; each
    /// call produces a fresh set of independent views.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::RangeViews;
    ///
    /// let bytes = bytes::Bytes::from(&b"abcdef"[..]);
    /// let views = RangeViews::from_ref(&bytes)
    ///     .add(3..6)
    ///     .add(0..3)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Request order is preserved (not sorted by offset).
    /// assert_eq!(&views[0][..], b"def");
    /// assert_eq!(&views[1][..], b"abc");
    /// ```
    pub fn build(&self) -> Result<Vec<Bytes>, RangeError> {
        let bytes = &self.bytes;
        let len = bytes.len();

        // Validate all boundaries before producing any view: a failure must
        // not return a partial group.
        for (request, range) in self.requests.iter().enumerate() {
            if range.start > range.end || range.end > len {
                return Err(RangeError {
                    request,
                    range: range.clone(),
                    len,
                });
            }
        }

        Ok(self
            .requests
            .iter()
            .map(|range| bytes.slice(range.clone()))
            .collect())
    }
}

/// An error returned by [`RangeViews::build`] when a range request cannot be
/// satisfied.
///
/// All boundaries are validated before any view is produced, so a build error
/// means the whole batch failed: no views are returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeError {
    /// The request index (request identity) of the first invalid range.
    ///
    /// This is the position the range occupied in the request order, which is
    /// also the position the view would have had in the output.
    pub request: usize,
    /// The invalid range, resolved to absolute offsets.
    pub range: Range<usize>,
    /// The length of the shared [`Bytes`] buffer.
    pub len: usize,
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "range request {} ({}..{}) is out of bounds for a buffer of length {}",
            self.request, self.range.start, self.range.end, self.len,
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RangeError {}

/// Resolves a [`RangeBounds`] value into an absolute [`Range`], substituting
/// buffer-relative defaults for unbounded endpoints.
fn resolve(range: impl RangeBounds<usize>, len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.checked_add(1).expect("range start out of range"),
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(&n) => n.checked_add(1).expect("range end out of range"),
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };

    start..end
}
