//! SSE 4.2 implementation of the SIMD traits, corresponding to
//! `target_feature = "sse4.2"`.

use std::mem;
use std::ops::{Add, Sub, Mul, Div, Neg, BitAnd, Not};

use traits::{SimdF32, SimdMask};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Sse42F32(__m128);

#[derive(Clone, Copy)]
pub struct Sse42Mask(__m128);

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_add_ps(a: __m128, b: __m128) -> __m128 {
    _mm_add_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_sub_ps(a: __m128, b: __m128) -> __m128 {
    _mm_sub_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_mul_ps(a: __m128, b: __m128) -> __m128 {
    _mm_mul_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_div_ps(a: __m128, b: __m128) -> __m128 {
    _mm_div_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_set1_ps(a: f32) -> __m128 {
    _mm_set1_ps(a)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_floor_ps(a: __m128) -> __m128 {
    _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_round_nearest_ps(a: __m128) -> __m128 {
    _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_andnot_ps(a: __m128, b: __m128) -> __m128 {
    _mm_andnot_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_and_ps(a: __m128, b: __m128) -> __m128 {
    _mm_and_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_setr_ps(a: f32, b: f32, c: f32, d: f32) -> __m128 {
    _mm_setr_ps(a, b, c, d)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_loadu_ps(p: *const f32) -> __m128 {
    _mm_loadu_ps(p)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_storeu_ps(p: *mut f32, a: __m128) {
    _mm_storeu_ps(p, a);
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_cmpeq_ps(a: __m128, b: __m128) -> __m128 {
    _mm_cmp_ps(a, b, _CMP_EQ_UQ)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_set1_epi32(a: i32) -> __m128i {
    _mm_set1_epi32(a)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_blendv_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_blendv_ps(a, b, c)
}

impl Add for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn add(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_add_ps(self.0, other.0))
        }
    }
}

impl Add<f32> for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn add(self, other: f32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_add_ps(self.0, sse42_set1_ps(other)))
        }
    }
}

impl Add<Sse42F32> for f32 {
    type Output = Sse42F32;
    #[inline]
    fn add(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_add_ps(sse42_set1_ps(self), other.0))
        }
    }
}

impl Sub for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn sub(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_sub_ps(self.0, other.0))
        }
    }
}

impl Sub<f32> for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn sub(self, other: f32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_sub_ps(self.0, sse42_set1_ps(other)))
        }
    }
}

impl Sub<Sse42F32> for f32 {
    type Output = Sse42F32;
    #[inline]
    fn sub(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_sub_ps(sse42_set1_ps(self), other.0))
        }
    }
}

impl Mul for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn mul(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_mul_ps(self.0, other.0))
        }
    }
}

impl Mul<f32> for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn mul(self, other: f32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_mul_ps(self.0, sse42_set1_ps(other)))
        }
    }
}

impl Mul<Sse42F32> for f32 {
    type Output = Sse42F32;
    #[inline]
    fn mul(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_mul_ps(sse42_set1_ps(self), other.0))
        }
    }    
}

impl Div for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn div(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_div_ps(self.0, other.0))
        }
    }
}

impl Div<f32> for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn div(self, other: f32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_div_ps(self.0, sse42_set1_ps(other)))
        }
    }
}

impl Div<Sse42F32> for f32 {
    type Output = Sse42F32;
    #[inline]
    fn div(self, other: Sse42F32) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_div_ps(sse42_set1_ps(self), other.0))
        }
    }    
}

impl Neg for Sse42F32 {
    type Output = Sse42F32;
    #[inline]
    fn neg(self) -> Sse42F32 {
        unsafe {
            Sse42F32(sse42_sub_ps(sse42_set1_ps(0.0), self.0))
        }
    }
}

impl From<Sse42F32> for __m128 {
    #[inline]
    fn from(x: Sse42F32) -> __m128 {
        x.0
    }
}

impl SimdF32 for Sse42F32 {
    type Raw = __m128;

    type Mask = Sse42Mask;

    #[inline]
    fn width(self) -> usize { 4 }

    #[inline]
    fn floor(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_floor_ps(self.0)) }
    }

    #[inline]
    fn round(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_round_nearest_ps(self.0)) }
    }

    #[inline]
    fn abs(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_andnot_ps(sse42_set1_ps(-0.0), self.0)) }
    }

    #[inline]
    fn splat(self, x: f32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_set1_ps(x)) }
    }

    #[inline]
    fn steps(self) -> Sse42F32 {
        unsafe { Sse42F32(sse42_setr_ps(0.0, 1.0, 2.0, 3.0)) }
    }

    #[inline]
    unsafe fn from_raw(raw: __m128) -> Sse42F32 {
        Sse42F32(raw)
    }

    #[inline]
    unsafe fn load(p: *const f32) -> Sse42F32 {
        Sse42F32(sse42_loadu_ps(p))
    }

    #[inline]
    unsafe fn store(self, p: *mut f32) {
        sse42_storeu_ps(p, self.0);
    }

    #[inline]
    unsafe fn create() -> Sse42F32 {
        Sse42F32(sse42_set1_ps(0.0))
    }

    #[inline]
    fn eq(self, other: Sse42F32) -> Sse42Mask {
        unsafe { Sse42Mask(sse42_cmpeq_ps(self.0, other.0)) }
    }
}

// Sse42Mask

impl From<Sse42Mask> for __m128 {
    #[inline]
    fn from(x: Sse42Mask) -> __m128 {
        x.0
    }
}

impl BitAnd for Sse42Mask {
    type Output = Sse42Mask;
    #[inline]
    fn bitand(self, other: Sse42Mask) -> Sse42Mask {
        unsafe { Sse42Mask(sse42_and_ps(self.0, other.0)) }
    }
}

impl Not for Sse42Mask {
    type Output = Sse42Mask;
    #[inline]
    fn not(self) -> Sse42Mask {
        unsafe { Sse42Mask(sse42_andnot_ps(self.0, mem::transmute(sse42_set1_epi32(-1)))) }
    }
}

impl SimdMask for Sse42Mask {
    type Raw = __m128;

    type F32 = Sse42F32;

    #[inline]
    fn select(self, a: Sse42F32, b: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_blendv_ps(b.0, a.0, self.0)) }
    }
}
