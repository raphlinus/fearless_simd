//! Generic traits for combining SIMD operations.

use traits::SimdF32;

pub trait SimdFnF32 {
    fn call<S: SimdF32>(&mut self, x: S) -> S;
}
