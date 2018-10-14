//! Fallback implementations of the SIMD traits.

use std::ptr;

use traits::{SimdF32, SimdMask};

impl SimdF32 for f32 {
    type Raw = f32;

    type Mask = u32;

    fn width(self) -> usize { 1 }

    fn round(self) -> f32 { f32::round(self) }

    fn abs(self) -> f32 { f32::abs(self) }

    fn splat(self, x: f32) -> f32 { x }

    fn steps(self) -> f32 { 0.0 }

    unsafe fn from_raw(raw: f32) -> f32 { raw }

    unsafe fn load(p: *const f32) -> f32 { ptr::read(p) }

    unsafe fn store(self, p: *mut f32) { ptr::write(p, self); }

    unsafe fn create() -> f32 { 0.0 }

    fn eq(self, other: f32) -> u32 {
        if self == other { !0 } else { 0 }
    }
}

impl SimdMask for u32 {
    type Raw = u32;
    type F32 = f32;

    fn select(self, a: f32, b: f32) -> f32 {
        if self & 0x80000000 != 0 { a } else { b }
    }
}
