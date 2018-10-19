//! Traits for SIMD operations.

use std::ops::{Add, Sub, Mul, Div, Neg, BitAnd};

pub trait SimdF32: Sized + Copy + Clone
    + Add<Self, Output=Self> + Add<f32, Output=Self>
    + Sub<Self, Output=Self> + Sub<f32, Output=Self>
    + Mul<Self, Output=Self> + Mul<f32, Output=Self>
    + Div<Self, Output=Self> + Mul<f32, Output=Self>
    + Neg<Output=Self>
    // The following would be convenient but run into limitations in the Rust
    // type system, which might be fixed at some point.
    // See Rust issue #23856 and a large number of related ones.
    /*
    where f32: Add<Self, Output=Self>,
    f32: Sub<Self, Output=Self>,
    f32: Mul<Self, Output=Self>,
    */
{
    type Raw: From<Self>;

    type Mask: SimdMask32<F32 = Self>;

    // Maybe doesn't need self?
    fn width(self) -> usize;

    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> Self;

    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> Self;

    /// Round a float to the nearest integer.
    ///
    /// The behavior on a tie is unspecified, and will be whatever is
    /// fastest on a given implementation. The ideal behavior is to round
    /// to the nearest even integer on tie; note that this is different
    /// than `f32::round`.
    ///
    /// See https://github.com/rust-lang/rust/issues/55107 for discussion.
    fn round(self) -> Self;

    /// Returns the absolute value of a number.
    fn abs(self) -> Self;

    /// Repeat a scalar in all lanes.
    ///
    /// Note: self is unused but is needed for safety.
    fn splat(self, x: f32) -> Self;

    /// Create SIMD that contains the lane number.
    ///
    /// For example, for 4 lanes, it is [0.0, 1.0, 2.0, 3.0].
    ///
    /// Note: self is unused but is needed for safety.
    fn steps(self) -> Self;

    /// Create from a raw value. Marked as unsafe because it requires that the
    /// corresponding target_feature is enabled.
    unsafe fn from_raw(raw: Self::Raw) -> Self;

    unsafe fn load(p: *const f32) -> Self;

    /// Load from a slice.
    ///
    /// # Panics
    ///
    /// If `slice.len() < Self::width()`.
    ///
    /// Note: self is unused but is needed for safety.
    fn from_slice(self, slice: &[f32]) -> Self {
        unsafe {
            assert!(slice.len() >= self.width());
            Self::load(slice.as_ptr())
        }
    }

    unsafe fn store(self, p: *mut f32);

    /// Write into a slice.
    ///
    /// # Panics
    ///
    /// If `slice.len() < Self::width()`.
    ///
    /// Note: self is unused but is needed for safety.
    fn write_to_slice(self, slice: &mut [f32]) {
        unsafe {
            assert!(slice.len() >= self.width());
            self.store(slice.as_mut_ptr());
        }
    }

    /// Create an instance (zero but value is usually ignored). Marked
    /// as unsafe because it requires that the corresponding target_feature
    /// is enabled.
    unsafe fn create() -> Self;

    fn eq(self, other: Self) -> Self::Mask;

    // TODO: other comparisons
}

/// A type compatible with an f32 simd value, representing a boolean in each lane.
pub trait SimdMask32: Sized + Copy + Clone
    + BitAnd<Self, Output=Self>
    where Self::Raw: From<Self>,
{
    type Raw;

    /// The corresponding compatible f32 type (with the same width).
    type F32: SimdF32<Mask = Self>;

    /// Select an element from `a` where the mask is true, and from `b`
    /// otherwise.
    fn select(self, a: Self::F32, b: Self::F32) -> Self::F32;
}

pub trait F32x4: Sized + Copy + Clone
    + Add<Self, Output=Self>
    + Mul + Mul<f32, Output=Self>
    where Self::Raw: From<Self>,
    // Again bitten by Rust #23856.
    /*
    [f32; 4]: From<Self>,
    */
{
    type Raw;

    /// Create an instance (zero but value is usually ignored). Marked
    /// as unsafe because it requires that the corresponding target_feature
    /// is enabled.
    unsafe fn create() -> Self;

    /// Create from a raw value. Marked as unsafe because it requires that the
    /// corresponding target_feature is enabled.
    unsafe fn from_raw(raw: Self::Raw) -> Self;

    /// Note: self is unused but is needed for safety.
    fn new(self, array: [f32; 4]) -> Self;

    fn as_vec(self) -> [f32; 4];
}
