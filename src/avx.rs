//! AVX implementation of the SIMD traits, corresponding to
//! `target_feature = "avx"`.

use std::mem;
use std::ops::{Add, Sub, Mul, Div, Neg, BitAnd, Not};

use traits::{SimdF32, SimdMask32};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct AvxF32(__m256);

#[derive(Clone, Copy)]
pub struct AvxMask32(__m256);

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_add_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_add_ps(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_sub_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_sub_ps(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_mul_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_mul_ps(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_div_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_div_ps(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_set1_ps(a: f32) -> __m256 {
    _mm256_set1_ps(a)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_floor_ps(a: __m256) -> __m256 {
    _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_ceil_ps(a: __m256) -> __m256 {
    _mm256_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_round_nearest_ps(a: __m256) -> __m256 {
    _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_andnot_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_andnot_ps(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_and_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_and_ps(a, b)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_setr_ps(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> __m256 {
    _mm256_setr_ps(a, b, c, d, e, f, g, h)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_loadu_ps(p: *const f32) -> __m256 {
    _mm256_loadu_ps(p)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_storeu_ps(p: *mut f32, a: __m256) {
    _mm256_storeu_ps(p, a);
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_cmpeq_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_set1_epi32(a: i32) -> __m256i {
    _mm256_set1_epi32(a)
}

#[inline]
#[target_feature(enable = "avx")]
unsafe fn avx_blendv_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_blendv_ps(a, b, c)
}

impl Add for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn add(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_add_ps(self.0, other.0))
        }
    }
}

impl Add<f32> for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn add(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(avx_add_ps(self.0, avx_set1_ps(other)))
        }
    }
}

impl Add<AvxF32> for f32 {
    type Output = AvxF32;
    #[inline]
    fn add(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_add_ps(avx_set1_ps(self), other.0))
        }
    }
}

impl Sub for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn sub(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_sub_ps(self.0, other.0))
        }
    }
}

impl Sub<f32> for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn sub(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(avx_sub_ps(self.0, avx_set1_ps(other)))
        }
    }
}

impl Sub<AvxF32> for f32 {
    type Output = AvxF32;
    #[inline]
    fn sub(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_sub_ps(avx_set1_ps(self), other.0))
        }
    }
}

impl Mul for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn mul(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_mul_ps(self.0, other.0))
        }
    }
}

impl Mul<f32> for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn mul(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(avx_mul_ps(self.0, avx_set1_ps(other)))
        }
    }
}

impl Mul<AvxF32> for f32 {
    type Output = AvxF32;
    #[inline]
    fn mul(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_mul_ps(avx_set1_ps(self), other.0))
        }
    }    
}

impl Div for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn div(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_div_ps(self.0, other.0))
        }
    }
}

impl Div<f32> for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn div(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(avx_div_ps(self.0, avx_set1_ps(other)))
        }
    }
}

impl Div<AvxF32> for f32 {
    type Output = AvxF32;
    #[inline]
    fn div(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(avx_div_ps(avx_set1_ps(self), other.0))
        }
    }    
}

impl Neg for AvxF32 {
    type Output = AvxF32;
    #[inline]
    fn neg(self) -> AvxF32 {
        unsafe {
            AvxF32(avx_sub_ps(avx_set1_ps(0.0), self.0))
        }
    }
}

impl From<AvxF32> for __m256 {
    #[inline]
    fn from(x: AvxF32) -> __m256 {
        x.0
    }
}

impl SimdF32 for AvxF32 {
    type Raw = __m256;

    type Mask = AvxMask32;

    #[inline]
    fn width(self) -> usize { 8 }

    #[inline]
    fn floor(self: AvxF32) -> AvxF32 {
        unsafe { AvxF32(avx_floor_ps(self.0)) }
    }

    #[inline]
    fn ceil(self: AvxF32) -> AvxF32 {
        unsafe { AvxF32(avx_ceil_ps(self.0)) }
    }

    #[inline]
    fn round(self: AvxF32) -> AvxF32 {
        unsafe { AvxF32(avx_round_nearest_ps(self.0)) }
    }

    #[inline]
    fn abs(self: AvxF32) -> AvxF32 {
        unsafe { AvxF32(avx_andnot_ps(avx_set1_ps(-0.0), self.0)) }
    }

    #[inline]
    fn splat(self, x: f32) -> AvxF32 {
        unsafe { AvxF32(avx_set1_ps(x)) }
    }

    #[inline]
    fn steps(self) -> AvxF32 {
        unsafe { AvxF32(avx_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)) }
    }

    #[inline]
    unsafe fn from_raw(raw: __m256) -> AvxF32 {
        AvxF32(raw)
    }

    #[inline]
    unsafe fn load(p: *const f32) -> AvxF32 {
        AvxF32(avx_loadu_ps(p))
    }

    #[inline]
    unsafe fn store(self, p: *mut f32) {
        avx_storeu_ps(p, self.0);
    }

    #[inline]
    unsafe fn create() -> AvxF32 {
        AvxF32(avx_set1_ps(0.0))
    }

    #[inline]
    fn eq(self, other: AvxF32) -> AvxMask32 {
        unsafe { AvxMask32(avx_cmpeq_ps(self.0, other.0)) }
    }
}

// AvxMask32

impl From<AvxMask32> for __m256 {
    #[inline]
    fn from(x: AvxMask32) -> __m256 {
        x.0
    }
}

impl BitAnd for AvxMask32 {
    type Output = AvxMask32;
    #[inline]
    fn bitand(self, other: AvxMask32) -> AvxMask32 {
        unsafe { AvxMask32(avx_and_ps(self.0, other.0)) }
    }
}

impl Not for AvxMask32 {
    type Output = AvxMask32;
    #[inline]
    fn not(self) -> AvxMask32 {
        unsafe { AvxMask32(avx_andnot_ps(self.0, mem::transmute(avx_set1_epi32(-1)))) }
    }
}

impl SimdMask32 for AvxMask32 {
    type Raw = __m256;

    type F32 = AvxF32;

    #[inline]
    fn select(self, a: AvxF32, b: AvxF32) -> AvxF32 {
        unsafe { AvxF32(avx_blendv_ps(b.0, a.0, self.0)) }
    }
}
