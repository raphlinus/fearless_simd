//! Fallback implementations of the SIMD traits.

use std::ptr;

use traits::{SimdF32, SimdMask};

impl SimdF32 for f32 {
    type Raw = f32;

    type Mask = u32;

    #[inline]
    fn width(self) -> usize { 1 }

    #[inline]
    fn floor(self) -> f32 { f32::floor(self) }

    #[inline]
    // See https://github.com/rust-lang/rust/issues/55107 for explanation
    // why I use `floor` here and not `round`.
    fn round(self) -> f32 { f32::floor(self + 0.5) }

    #[inline]
    fn abs(self) -> f32 { f32::abs(self) }

    #[inline]
    fn splat(self, x: f32) -> f32 { x }

    #[inline]
    fn steps(self) -> f32 { 0.0 }

    #[inline]
    unsafe fn from_raw(raw: f32) -> f32 { raw }

    #[inline]
    unsafe fn load(p: *const f32) -> f32 { ptr::read(p) }

    #[inline]
    unsafe fn store(self, p: *mut f32) { ptr::write(p, self); }

    #[inline]
    unsafe fn create() -> f32 { 0.0 }

    #[inline]
    fn eq(self, other: f32) -> u32 {
        if self == other { !0 } else { 0 }
    }
}

impl SimdMask for u32 {
    type Raw = u32;
    type F32 = f32;

    #[inline]
    fn select(self, a: f32, b: f32) -> f32 {
        if self & 0x80000000 != 0 { a } else { b }
    }
}
