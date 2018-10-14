//! AVX implementation of the SIMD traits, corresponding to
//! `target_feature = "avx"`.

use std::mem;
use std::ops::{Add, Sub, Mul, Div, Neg, BitAnd, Not};

use traits::{SimdF32, SimdMask};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct AvxF32(__m256);

#[derive(Clone, Copy)]
pub struct AvxMask(__m256);

impl Add for AvxF32 {
    type Output = AvxF32;
    fn add(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_add_ps(self.0, other.0))
        }
    }
}

impl Add<f32> for AvxF32 {
    type Output = AvxF32;
    fn add(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_add_ps(self.0, _mm256_set1_ps(other)))
        }
    }
}

impl Add<AvxF32> for f32 {
    type Output = AvxF32;
    fn add(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_add_ps(_mm256_set1_ps(self), other.0))
        }
    }
}

impl Sub for AvxF32 {
    type Output = AvxF32;
    fn sub(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_sub_ps(self.0, other.0))
        }
    }
}

impl Sub<f32> for AvxF32 {
    type Output = AvxF32;
    fn sub(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_sub_ps(self.0, _mm256_set1_ps(other)))
        }
    }
}

impl Sub<AvxF32> for f32 {
    type Output = AvxF32;
    fn sub(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_sub_ps(_mm256_set1_ps(self), other.0))
        }
    }
}

impl Mul for AvxF32 {
    type Output = AvxF32;
    fn mul(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_mul_ps(self.0, other.0))
        }
    }
}

impl Mul<f32> for AvxF32 {
    type Output = AvxF32;
    fn mul(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_mul_ps(self.0, _mm256_set1_ps(other)))
        }
    }
}

impl Mul<AvxF32> for f32 {
    type Output = AvxF32;
    fn mul(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_mul_ps(_mm256_set1_ps(self), other.0))
        }
    }    
}

impl Div for AvxF32 {
    type Output = AvxF32;
    fn div(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_div_ps(self.0, other.0))
        }
    }
}

impl Div<f32> for AvxF32 {
    type Output = AvxF32;
    fn div(self, other: f32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_div_ps(self.0, _mm256_set1_ps(other)))
        }
    }
}

impl Div<AvxF32> for f32 {
    type Output = AvxF32;
    fn div(self, other: AvxF32) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_div_ps(_mm256_set1_ps(self), other.0))
        }
    }    
}

impl Neg for AvxF32 {
    type Output = AvxF32;
    fn neg(self) -> AvxF32 {
        unsafe {
            AvxF32(_mm256_sub_ps(_mm256_set1_ps(0.0), self.0))
        }
    }
}

impl From<AvxF32> for __m256 {
    fn from(x: AvxF32) -> __m256 {
        x.0
    }
}

impl SimdF32 for AvxF32 {
    type Raw = __m256;

    type Mask = AvxMask;

    fn width(self) -> usize { 8 }

    fn round(self: AvxF32) -> AvxF32 {
        unsafe { AvxF32(_mm256_round_ps(self.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) }
    }

    fn abs(self: AvxF32) -> AvxF32 {
        unsafe { AvxF32(_mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0)) }
    }

    fn splat(self, x: f32) -> AvxF32 {
        unsafe { AvxF32(_mm256_set1_ps(x)) }
    }

    fn steps(self) -> AvxF32 {
        unsafe { AvxF32(_mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)) }
    }

    unsafe fn from_raw(raw: __m256) -> AvxF32 {
        AvxF32(raw)
    }

    unsafe fn load(p: *const f32) -> AvxF32 {
        AvxF32(_mm256_loadu_ps(p))
    }

    unsafe fn store(self, p: *mut f32) {
        _mm256_storeu_ps(p, self.0);
    }

    unsafe fn create() -> AvxF32 {
        AvxF32(_mm256_set1_ps(0.0))
    }

    fn eq(self, other: AvxF32) -> AvxMask {
        unsafe { AvxMask(_mm256_cmp_ps(self.0, other.0, _CMP_EQ_UQ)) }
    }
}

// AvxMask

impl From<AvxMask> for __m256 {
    fn from(x: AvxMask) -> __m256 {
        x.0
    }
}

impl BitAnd for AvxMask {
    type Output = AvxMask;
    fn bitand(self, other: AvxMask) -> AvxMask {
        unsafe { AvxMask(_mm256_and_ps(self.0, other.0)) }
    }
}

impl Not for AvxMask {
    type Output = AvxMask;
    fn not(self) -> AvxMask {
        unsafe { AvxMask(_mm256_andnot_ps(self.0, mem::transmute(_mm256_set1_epi32(-1)))) }
    }
}

impl SimdMask for AvxMask {
    type Raw = __m256;

    type F32 = AvxF32;

    fn select(self, a: AvxF32, b: AvxF32) -> AvxF32 {
        unsafe { AvxF32(_mm256_blendv_ps(b.0, a.0, self.0)) }
    }
}
