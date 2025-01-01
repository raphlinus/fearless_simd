// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to SSSE3 intrinsics.

use crate::impl_macros::delegate;
use core::arch::x86_64::*;

/// A token for SSSE3 intrinsics on x86_64.
#[derive(Clone, Copy, Debug)]
pub struct Ssse3 {
    _private: (),
}

impl Ssse3 {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    pub unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }

    delegate! { core::arch::x86_64:
        fn _mm_abs_epi8(a: __m128i) -> __m128i;
        fn _mm_abs_epi16(a: __m128i) -> __m128i;
        fn _mm_abs_epi32(a: __m128i) -> __m128i;
        fn _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_alignr_epi8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_hadd_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_hadds_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_hadd_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_hsub_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_hsubs_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_hsub_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_maddubs_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sign_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sign_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sign_epi32(a: __m128i, b: __m128i) -> __m128i;
    }
}
