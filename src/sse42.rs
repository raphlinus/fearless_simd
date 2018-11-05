//! SSE 4.2 implementation of the SIMD traits, corresponding to
//! `target_feature = "sse4.2"`.

use std::mem;
use std::ops::{Add, Sub, Mul, Div, Neg, BitAnd, Not, Deref};

use traits::{SimdF32, SimdMask32, F32x4};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Sse42F32(__m128);

#[derive(Clone, Copy)]
pub struct Sse42Mask32(__m128);

#[derive(Clone, Copy)]
pub struct Sse42F32x4(__m128);

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
unsafe fn sse42_ceil_ps(a: __m128) -> __m128 {
    _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_round_nearest_ps(a: __m128) -> __m128 {
    _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_min_ps(a: __m128, b: __m128) -> __m128 {
    _mm_min_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_max_ps(a: __m128, b: __m128) -> __m128 {
    _mm_max_ps(a, b)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_rcp_ps(a: __m128) -> __m128 {
    _mm_rcp_ps(a)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_rsqrt_ps(a: __m128) -> __m128 {
    _mm_rsqrt_ps(a)
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_sqrt_ps(a: __m128) -> __m128 {
    _mm_sqrt_ps(a)
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

    type Mask = Sse42Mask32;

    #[inline]
    fn width(self) -> usize { 4 }

    #[inline]
    fn floor(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_floor_ps(self.0)) }
    }

    #[inline]
    fn ceil(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_ceil_ps(self.0)) }
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
    fn min(self: Sse42F32, b: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_min_ps(self.0, b.0)) }
    }

    #[inline]
    fn max(self: Sse42F32, b: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_max_ps(self.0, b.0)) }
    }

    #[inline]
    fn recip11(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_rcp_ps(self.0)) }
    }

    #[inline]
    fn recip22(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32( {
            let est = sse42_rcp_ps(self.0);
            let muls = sse42_mul_ps(self.0, sse42_mul_ps(est, est));
            sse42_sub_ps(sse42_add_ps(est, est), muls)
        })}
    }

    #[inline]
    fn recip(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_div_ps(sse42_set1_ps(1.0), self.0)) }
    }

    #[inline]
    fn rsqrt11(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_rsqrt_ps(self.0)) }
    }

    #[inline]
    fn rsqrt22(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32( {
            let est = sse42_rsqrt_ps(self.0);
            let r_est = sse42_mul_ps(self.0, est);
            let half_est = sse42_mul_ps(sse42_set1_ps(0.5), est);
            let muls = sse42_mul_ps(r_est, est);
            let three_minus_muls = sse42_sub_ps(sse42_set1_ps(3.0), muls);
            sse42_mul_ps(half_est, three_minus_muls)
        })}
    }

    #[inline]
    fn rsqrt(self: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_div_ps(sse42_set1_ps(1.0), sse42_sqrt_ps(self.0))) }
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
    fn eq(self, other: Sse42F32) -> Sse42Mask32 {
        unsafe { Sse42Mask32(sse42_cmpeq_ps(self.0, other.0)) }
    }
}

// Sse42Mask32

impl From<Sse42Mask32> for __m128 {
    #[inline]
    fn from(x: Sse42Mask32) -> __m128 {
        x.0
    }
}

impl BitAnd for Sse42Mask32 {
    type Output = Sse42Mask32;
    #[inline]
    fn bitand(self, other: Sse42Mask32) -> Sse42Mask32 {
        unsafe { Sse42Mask32(sse42_and_ps(self.0, other.0)) }
    }
}

impl Not for Sse42Mask32 {
    type Output = Sse42Mask32;
    #[inline]
    fn not(self) -> Sse42Mask32 {
        unsafe { Sse42Mask32(sse42_andnot_ps(self.0, mem::transmute(sse42_set1_epi32(-1)))) }
    }
}

impl SimdMask32 for Sse42Mask32 {
    type Raw = __m128;

    type F32 = Sse42F32;

    #[inline]
    fn select(self, a: Sse42F32, b: Sse42F32) -> Sse42F32 {
        unsafe { Sse42F32(sse42_blendv_ps(b.0, a.0, self.0)) }
    }
}

// F32x4

impl From<Sse42F32x4> for __m128 {
    #[inline]
    fn from(x: Sse42F32x4) -> __m128 {
        x.0
    }
}

impl From<Sse42F32x4> for [f32; 4] {
    #[inline]
    fn from(x: Sse42F32x4) -> [f32; 4] {
        x.as_vec()
    }
}

impl Deref for Sse42F32x4 {
    type Target = [f32; 4];
    #[inline]
    fn deref(&self) -> &[f32; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl Add for Sse42F32x4 {
    type Output = Sse42F32x4;

    #[inline]
    fn add(self, other: Sse42F32x4) -> Sse42F32x4 {
        unsafe { Sse42F32x4(sse42_add_ps(self.0, other.0)) }
    }
}

impl Mul for Sse42F32x4 {
    type Output = Sse42F32x4;

    #[inline]
    fn mul(self, other: Sse42F32x4) -> Sse42F32x4 {
        unsafe { Sse42F32x4(sse42_mul_ps(self.0, other.0)) }
    }
}

impl Mul<f32> for Sse42F32x4 {
    type Output = Sse42F32x4;

    #[inline]
    fn mul(self, other: f32) -> Sse42F32x4 {
        unsafe { Sse42F32x4(sse42_mul_ps(self.0, sse42_set1_ps(other))) }
    }
}

impl F32x4 for Sse42F32x4 {
    type Raw = __m128;

    #[inline]
    unsafe fn from_raw(raw: __m128) -> Sse42F32x4 {
        Sse42F32x4(raw)
    }

    #[inline]
    unsafe fn create() -> Sse42F32x4 {
        Sse42F32x4(sse42_set1_ps(0.0))
    }

    #[inline]
    fn new(self, array: [f32; 4]) -> Sse42F32x4 {
        union U {
            array: [f32; 4],
            xmm: __m128,
        }
        unsafe { Sse42F32x4(U { array }.xmm) }
    }

    #[inline]
    fn as_vec(self) -> [f32; 4] {
        union U {
            array: [f32; 4],
            xmm: __m128,
        }
        unsafe { U { xmm: self.0 }.array }
    }
}