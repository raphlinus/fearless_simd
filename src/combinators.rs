//! Generic traits for combining SIMD operations.

use traits::SimdF32;

/// A trait representing f32 -> f32 function that can be computed
/// using simd.
pub trait SimdFnF32 {
    /// Compute one simd chunk of the function.
    ///
    /// Note: always annotate the implementation with `#[inline]`.
    /// If not, performance will suffer, and it triggers compiler
    /// bugs.
    fn call<S: SimdF32>(&mut self, x: S) -> S;
}
