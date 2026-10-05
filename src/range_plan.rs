//! Explicit multi-range view plans over a shared [`Bytes`] storage.
//!
//! A [`RangePlan`] describes a batch of half-open `[start, end)` ranges that
//! are later turned into [`Bytes`] views. All ranges in the plan share the
//! same underlying storage of the source buffer: minting a view never copies
//! bytes and never merges overlapping ranges into one. Each requested range
//! yields exactly one view, and views are returned **in request order** (the
//! identity of the request), not sorted by offset.
//!
//! # Validation
//!
//! Every boundary is validated *before* any view is created. An empty range
//! (`start == end`) is legal and produces an empty view that still occupies
//! its request slot. If any range is inverted (`start > end`) or extends past
//! the end of the buffer, the whole batch fails and **no** views are
//! returned — a partial batch is never produced.
//!
//! # Owned vs. borrowed relationship
//!
//! The returned views are *owned* [`Bytes`] handles that *borrow* (share,
//! via refcount) the underlying storage of the source buffer. The storage is
//! kept alive as long as any view is alive, even if the original source
//! handle is dropped. Views are immutable: a view can only be minted from an
//! immutable [`Bytes`], or from a [`BytesMut`] by **consuming** it through
//! [`BytesMut::freeze_views`] (or by calling [`BytesMut::freeze`] first).
//! There is no API that mints views from a `&mut BytesMut` borrow, so the
//! freeze semantics can never be bypassed — no mutable handle can alias a
//! live view.
//!
//! # Example
//!
//! Four ranges over one buffer, including an empty range and an overlap:
//!
//! ```
//! use bytes::{Bytes, RangePlan};
//!
//! let data = Bytes::from_static(b"abcdefghijklmnopqrstuvwxyz");
//!
//! let mut plan = RangePlan::new();
//! plan.push(0, 4).unwrap();   // request 0: "abcd"
//! plan.push(4, 4).unwrap();   // request 1: empty view, still a request identity
//! plan.push(2, 6).unwrap();   // request 2: "cdef", overlaps request 0, not merged
//! plan.push(22, 26).unwrap(); // request 3: "wxyz"
//!
//! let views = data.view_plan(&plan).unwrap();
//! assert_eq!(&views[0][..], b"abcd");
//! assert_eq!(&views[1][..], b"");
//! assert_eq!(&views[2][..], b"cdef");
//! assert_eq!(&views[3][..], b"wxyz");
//!
//! // The original handle may be dropped; the views keep the storage alive.
//! drop(data);
//! assert_eq!(&views[0][..], b"abcd");
//! ```

use alloc::vec::Vec;
use core::fmt;

use crate::{Bytes, BytesMut};

/// A validated batch of `[start, end)` ranges to be minted as [`Bytes`] views.
///
/// See the [module-level documentation](self) for the full contract.
#[derive(Debug, Clone, Default)]
pub struct RangePlan {
    /// Half-open ranges in request order. Invariant: `start <= end`.
    ranges: Vec<(usize, usize)>,
}

/// Why a range request was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangePlanErrorKind {
    /// `start > end`: the range is inverted.
    Inverted,
    /// `end` exceeds the length of the target buffer.
    OutOfBounds,
}

/// Error rejecting a range request. Carries the request identity (`index`)
/// of the offending range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangePlanError {
    index: usize,
    start: usize,
    end: usize,
    len: usize,
    kind: RangePlanErrorKind,
}

impl RangePlanError {
    fn inverted(index: usize, start: usize, end: usize) -> Self {
        RangePlanError {
            index,
            start,
            end,
            len: 0,
            kind: RangePlanErrorKind::Inverted,
        }
    }

    fn out_of_bounds(index: usize, start: usize, end: usize, len: usize) -> Self {
        RangePlanError {
            index,
            start,
            end,
            len,
            kind: RangePlanErrorKind::OutOfBounds,
        }
    }

    /// Position of the rejected range within the plan (its request identity).
    pub fn index(&self) -> usize {
        self.index
    }

    /// The rejected range start.
    pub fn start(&self) -> usize {
        self.start
    }

    /// The rejected range end.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Length of the target buffer the range was checked against. Only
    /// meaningful when [`kind`](Self::kind) is
    /// [`RangePlanErrorKind::OutOfBounds`].
    pub fn len(&self) -> usize {
        self.len
    }

    /// Which rule the range violated.
    pub fn kind(&self) -> RangePlanErrorKind {
        self.kind
    }
}

impl fmt::Display for RangePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            RangePlanErrorKind::Inverted => write!(
                f,
                "range request {} is inverted: start {} > end {}",
                self.index, self.start, self.end
            ),
            RangePlanErrorKind::OutOfBounds => write!(
                f,
                "range request {} [{}..{}) exceeds buffer length {}",
                self.index, self.start, self.end, self.len
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RangePlanError {}

impl RangePlan {
    /// Creates an empty plan.
    pub fn new() -> Self {
        RangePlan { ranges: Vec::new() }
    }

    /// Creates an empty plan with room for `capacity` ranges.
    pub fn with_capacity(capacity: usize) -> Self {
        RangePlan {
            ranges: Vec::with_capacity(capacity),
        }
    }

    /// Builds a plan from `(start, end)` pairs, validating each one.
    ///
    /// Empty ranges (`start == end`) are legal. Returns the identity of the
    /// first inverted range, if any.
    pub fn from_ranges<I>(ranges: I) -> Result<Self, RangePlanError>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let mut plan = RangePlan::new();
        for (start, end) in ranges {
            plan.push(start, end)?;
        }
        Ok(plan)
    }

    /// Appends the half-open range `[start, end)` to the plan.
    ///
    /// Empty ranges (`start == end`) are legal and yield empty views that
    /// keep their request identity. Bounds against a concrete buffer are
    /// checked later, when the plan is applied.
    pub fn push(&mut self, start: usize, end: usize) -> Result<(), RangePlanError> {
        if start > end {
            return Err(RangePlanError::inverted(self.ranges.len(), start, end));
        }
        self.ranges.push((start, end));
        Ok(())
    }

    /// Number of requested ranges.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the plan requests no ranges. Applying an empty plan succeeds
    /// and yields an empty batch.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The requested ranges, in request order.
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// Validates every range against `len` before any view is minted.
    fn check_all(&self, len: usize) -> Result<(), RangePlanError> {
        for (index, &(start, end)) in self.ranges.iter().enumerate() {
            debug_assert!(start <= end, "RangePlan invariant: start <= end");
            if end > len {
                return Err(RangePlanError::out_of_bounds(index, start, end, len));
            }
        }
        Ok(())
    }
}

impl Bytes {
    /// Mints one view per range in `plan`, all sharing this buffer's storage.
    ///
    /// The returned vector is indexed by **request identity**: `views[i]`
    /// corresponds to the `i`-th range pushed into the plan, regardless of
    /// offset order. Overlapping ranges are neither copied nor merged — each
    /// gets its own zero-copy view over the shared storage.
    ///
    /// All boundaries are validated first; if any range is out of bounds the
    /// whole batch fails with `Err` and no views are produced.
    pub fn view_plan(&self, plan: &RangePlan) -> Result<Vec<Bytes>, RangePlanError> {
        // Pass 1: validate every boundary up front so a failure can never
        // yield a partial batch.
        plan.check_all(self.len())?;
        // Pass 2: mint views in request order.
        Ok(plan
            .ranges
            .iter()
            .map(|&(start, end)| self.slice(start..end))
            .collect())
    }
}

impl BytesMut {
    /// Freezes this buffer and mints one [`Bytes`] view per range in `plan`.
    ///
    /// This consumes the `BytesMut`, so freeze semantics cannot be bypassed:
    /// after a successful call no mutable handle to the storage exists, and
    /// the returned views share the frozen storage without copying.
    ///
    /// All boundaries are validated *before* freezing. On failure the buffer
    /// is handed back unchanged together with the error, and no views are
    /// produced.
    pub fn freeze_views(
        self,
        plan: &RangePlan,
    ) -> Result<Vec<Bytes>, (BytesMut, RangePlanError)> {
        if let Err(err) = plan.check_all(self.len()) {
            return Err((self, err));
        }
        let frozen = self.freeze();
        Ok(plan
            .ranges
            .iter()
            .map(|&(start, end)| frozen.slice(start..end))
            .collect())
    }
}
