// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to FMA intrinsics.

use crate::impl_macros::delegate;
use core::arch::x86_64::*;

/// A token for FMA intrinsics on x86_64.
#[derive(Clone, Copy, Debug)]
pub struct Fma {
    _private: (),
}

impl Fma {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    #[inline]
    pub unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }

    delegate! { core::arch::x86_64:
        fn _mm_fmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm256_fmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm_fmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm256_fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm_fmadd_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm_fmadd_ss(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm_fmaddsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm256_fmaddsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm_fmaddsub_ps(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm256_fmaddsub_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm_fmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm256_fmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm_fmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm256_fmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm_fmsub_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm_fmsub_ss(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm_fmsubadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm256_fmsubadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm_fmsubadd_ps(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm256_fmsubadd_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm_fnmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm256_fnmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm_fnmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm256_fnmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm_fnmadd_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm_fnmadd_ss(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm_fnmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm256_fnmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm_fnmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128;
        fn _mm256_fnmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm_fnmsub_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
        fn _mm_fnmsub_ss(a: __m128, b: __m128, c: __m128) -> __m128;
    }
}
