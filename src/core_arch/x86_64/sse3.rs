// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to SSE3 intrinsics.

use crate::impl_macros::delegate;
use core::arch::x86_64::*;

/// A token for SSE3 intrinsics on x86_64.
#[derive(Clone, Copy, Debug)]
pub struct Sse3 {
    _private: (),
}

impl Sse3 {
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
        fn _mm_addsub_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_addsub_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_hadd_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_hadd_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_hsub_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_hsub_ps(a: __m128, b: __m128) -> __m128;
        unsafe fn _mm_lddqu_si128(mem_addr: *const __m128i) -> __m128i;
        fn _mm_movedup_pd(a: __m128d) -> __m128d;
        unsafe fn _mm_loaddup_pd(mem_addr: *const f64) -> __m128d;
        fn _mm_movehdup_ps(a: __m128) -> __m128;
        fn _mm_moveldup_ps(a: __m128) -> __m128;
    }
}
