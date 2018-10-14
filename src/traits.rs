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

    type Mask: SimdMask<F32 = Self>;

    // Maybe doesn't need self?
    fn width(self) -> usize;

    fn round(self) -> Self;

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

pub trait SimdMask: Sized + Copy + Clone
    + BitAnd<Self, Output=Self>
    where Self::Raw: From<Self>,
{
    type Raw;

    type F32: SimdF32<Mask = Self>;

    fn select(self, a: Self::F32, b: Self::F32) -> Self::F32;
}
