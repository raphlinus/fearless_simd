// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the Avx2 level.

use crate::{
    f32x4, f32x8,
    impl_macros::{impl_op, impl_simd_from_into},
    mask32x4, mask32x8,
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
impl_simd_from_into!(f32x8, __m256);
impl_simd_from_into!(mask32x8, __m256i);

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

    impl_op!(simd_eq_f32x4(a: f32x4, b: f32x4) -> mask32x4 = sse2._mm_castps_si128(sse._mm_cmpeq_ps));
    impl_op!(simd_ne_f32x4(a: f32x4, b: f32x4) -> mask32x4 = sse2._mm_castps_si128(sse._mm_cmpneq_ps));
    impl_op!(simd_lt_f32x4(a: f32x4, b: f32x4) -> mask32x4 = sse2._mm_castps_si128(sse._mm_cmplt_ps));
    impl_op!(simd_le_f32x4(a: f32x4, b: f32x4) -> mask32x4 = sse2._mm_castps_si128(sse._mm_cmple_ps));
    impl_op!(simd_ge_f32x4(a: f32x4, b: f32x4) -> mask32x4 = sse2._mm_castps_si128(sse._mm_cmpge_ps));
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

    #[inline(always)]
    fn splat_f32x8(self, val: f32) -> f32x8<Self> {
        self.avx._mm256_set1_ps(val).simd_into(self)
    }
    impl_op!(add_f32x8(a: f32x8, b: f32x8) -> f32x8 = avx._mm256_add_ps);
    impl_op!(sub_f32x8(a: f32x8, b: f32x8) -> f32x8 = avx._mm256_sub_ps);
    impl_op!(mul_f32x8(a: f32x8, b: f32x8) -> f32x8 = avx._mm256_mul_ps);
    impl_op!(div_f32x8(a: f32x8, b: f32x8) -> f32x8 = avx._mm256_div_ps);
    impl_op!(mul_add_f32x8(a: f32x8, b: f32x8, c: f32x8) -> f32x8 = fma._mm256_fmadd_ps);

    impl_op!(simd_eq_f32x8(a: f32x8, b: f32x8) -> mask32x8
        = avx._mm256_castps_si256(avx._mm256_cmp_ps<_CMP_EQ_OQ>));
    impl_op!(simd_ne_f32x8(a: f32x8, b: f32x8) -> mask32x8
        = avx._mm256_castps_si256(avx._mm256_cmp_ps<_CMP_NEQ_OQ>));
    impl_op!(simd_lt_f32x8(a: f32x8, b: f32x8) -> mask32x8
        = avx._mm256_castps_si256(avx._mm256_cmp_ps<_CMP_LT_OQ>));
    impl_op!(simd_le_f32x8(a: f32x8, b: f32x8) -> mask32x8
        = avx._mm256_castps_si256(avx._mm256_cmp_ps<_CMP_LE_OQ>));
    impl_op!(simd_ge_f32x8(a: f32x8, b: f32x8) -> mask32x8
        = avx._mm256_castps_si256(avx._mm256_cmp_ps<_CMP_GE_OQ>));
    impl_op!(simd_gt_f32x8(a: f32x8, b: f32x8) -> mask32x8
        = avx._mm256_castps_si256(avx._mm256_cmp_ps<_CMP_GT_OQ>));
    impl_op!(select_f32x8(a: mask32x8, b: f32x8, c: f32x8) -> f32x8
        = avx._mm256_blendv_ps(c, b, avx._mm256_castsi256_ps(a)));
    impl_op!(sqrt_f32x8(a: f32x8) -> f32x8 = avx._mm256_sqrt_ps);

    #[inline(always)]
    fn abs_f32x8(self, a: f32x8<Self>) -> f32x8<Self> {
        let sign_mask = self
            .avx
            ._mm256_castsi256_ps(self.avx._mm256_set1_epi32(0x7fff_ffff));
        self.avx._mm256_and_ps(sign_mask, a.into()).simd_into(self)
    }

    #[inline(always)]
    fn copysign_f32x8(self, a: f32x8<Self>, b: f32x8<Self>) -> f32x8<Self> {
        let sign_mask = self
            .avx
            ._mm256_castsi256_ps(self.avx._mm256_set1_epi32(-0x8000_0000));
        self.avx
            ._mm256_or_ps(
                self.avx._mm256_and_ps(sign_mask, b.into()),
                self.avx._mm256_andnot_ps(sign_mask, a.into()),
            )
            .simd_into(self)
    }

    impl_op!(combine_f32x4(a: f32x4, b: f32x4) -> f32x8 = avx._mm256_set_m128);

    #[inline(always)]
    fn split_f32x8(self, a: f32x8<Self>) -> (f32x4<Self>, f32x4<Self>) {
        let a1 = a.into();
        let b0 = self.avx._mm256_castps256_ps128(a1);
        let b1 = self.avx._mm256_extractf128_ps::<1>(a1);
        (b0.simd_into(self), b1.simd_into(self))
    }
}
