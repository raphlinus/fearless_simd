// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the Avx2 level.

use crate::{
    f32x4,
    impl_macros::{impl_op, impl_simd_from_into},
    mask32x4,
    seal::Seal,
    Simd, SimdFrom, SimdInto,
};
use core::arch::x86_64::*;

use super::Level;

/// The SIMD token for the "avx2" level.
///
/// This is short for the "x86-64-v3 microarchitecture level". In this level, the
/// following target_features are enabled: "avx2", "bmi2", "f16c", "fma", "lzcnt".

/// The SIMD token for the "avx2" level.
///
/// This is short for the "x86-64-v3 microarchitecture level". In this level, the
/// following target_features are enabled: "avx2", "bmi2", "f16c", "fma", "lzcnt".
#[derive(Clone, Copy, Debug)]
pub struct Avx2 {
    pub sse: crate::core_arch::x86_64::Sse,
    pub sse2: crate::core_arch::x86_64::Sse2,
    pub sse3: crate::core_arch::x86_64::Sse3,
    pub ssse3: crate::core_arch::x86_64::Ssse3,
    pub sse4_1: crate::core_arch::x86_64::Sse4_1,
    pub sse4_2: crate::core_arch::x86_64::Sse4_2,
    pub avx: crate::core_arch::x86_64::Avx,
    pub avx2: crate::core_arch::x86_64::Avx2,
    pub fma: crate::core_arch::x86_64::Fma,
}

impl Avx2 {
    #[inline]
    pub unsafe fn new_unchecked() -> Self {
        Avx2 {
            sse: crate::core_arch::x86_64::Sse::new_unchecked(),
            sse2: crate::core_arch::x86_64::Sse2::new_unchecked(),
            sse3: crate::core_arch::x86_64::Sse3::new_unchecked(),
            ssse3: crate::core_arch::x86_64::Ssse3::new_unchecked(),
            sse4_1: crate::core_arch::x86_64::Sse4_1::new_unchecked(),
            sse4_2: crate::core_arch::x86_64::Sse4_2::new_unchecked(),
            avx: crate::core_arch::x86_64::Avx::new_unchecked(),
            avx2: crate::core_arch::x86_64::Avx2::new_unchecked(),
            fma: crate::core_arch::x86_64::Fma::new_unchecked(),
        }
    }
}

impl_simd_from_into!(f32x4, __m128);
impl_simd_from_into!(mask32x4, __m128i);

impl Seal for Avx2 {}

impl Simd for Avx2 {
    #[inline(always)]
    fn level(self) -> Level {
        Level::Avx2(self)
    }

    #[inline]
    fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
        #[target_feature(enable = "avx2,bmi2,f16c,fma,lzcnt")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn vectorize_avx2<F: FnOnce() -> R, R>(f: F) -> R {
            f()
        }
        unsafe { vectorize_avx2(f) }
    }

    #[inline(always)]
    fn splat_f32x4(self, val: f32) -> f32x4<Self> {
        self.sse._mm_set1_ps(val).simd_into(self)
    }

    impl_op!(add_f32x4(a: f32x4, b: f32x4) -> f32x4 = sse._mm_add_ps);
    impl_op!(sub_f32x4(a: f32x4, b: f32x4) -> f32x4 = sse._mm_sub_ps);
    impl_op!(mul_f32x4(a: f32x4, b: f32x4) -> f32x4 = sse._mm_mul_ps);
    impl_op!(div_f32x4(a: f32x4, b: f32x4) -> f32x4 = sse._mm_div_ps);
    impl_op!(mul_add_f32x4(a: f32x4, b: f32x4, c: f32x4) -> f32x4 = fma._mm_fmadd_ps);

    impl_op!(simd_gt_f32x4(a: f32x4, b: f32x4) -> mask32x4 = sse2._mm_castps_si128(sse._mm_cmpgt_ps));
    impl_op!(select_f32x4(a: mask32x4, b: f32x4, c: f32x4) -> f32x4
        = sse4_1._mm_blendv_ps(c, b, sse2._mm_castsi128_ps(a)));
    impl_op!(sqrt_f32x4(a: f32x4) -> f32x4 = sse._mm_sqrt_ps);

    #[inline(always)]
    fn abs_f32x4(self, a: f32x4<Self>) -> f32x4<Self> {
        let sign_mask = self
            .sse2
            ._mm_castsi128_ps(self.sse2._mm_set1_epi32(0x7fff_ffff));
        self.sse._mm_and_ps(sign_mask, a.into()).simd_into(self)
    }

    #[inline(always)]
    fn copysign_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let sign_mask = self
            .sse2
            ._mm_castsi128_ps(self.sse2._mm_set1_epi32(-0x8000_0000));
        self.sse
            ._mm_or_ps(
                self.sse._mm_and_ps(sign_mask, b.into()),
                self.sse._mm_andnot_ps(sign_mask, a.into()),
            )
            .simd_into(self)
    }
}
