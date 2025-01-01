// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to SSE4.2 intrinsics.

use crate::impl_macros::delegate;
use core::arch::x86_64::*;

/// A token for SSE4.2 intrinsics on x86_64.
#[derive(Clone, Copy, Debug)]
pub struct Sse4_2 {
    _private: (),
}

impl Sse4_2 {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    pub unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }

    delegate! { core::arch::x86_64:
        fn _mm_cmpistrm<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpistri<const IMM8: i32>(a: __m128i, b: __m128i) -> i32;
        fn _mm_cmpistrz<const IMM8: i32>(a: __m128i, b: __m128i) -> i32;
        fn _mm_cmpistrc<const IMM8: i32>(a: __m128i, b: __m128i) -> i32;
        fn _mm_cmpistrs<const IMM8: i32>(a: __m128i, b: __m128i) -> i32;
        fn _mm_cmpistro<const IMM8: i32>(a: __m128i, b: __m128i) -> i32;
        fn _mm_cmpistra<const IMM8: i32>(a: __m128i, b: __m128i) -> i32;
        fn _mm_cmpestrm<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> __m128i;
        fn _mm_cmpestri<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32;
        fn _mm_cmpestrz<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32;
        fn _mm_cmpestrc<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32;
        fn _mm_cmpestrs<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32;
        fn _mm_cmpestro<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32;
        fn _mm_cmpestra<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32;
        fn _mm_crc32_u8(crc: u32, v: u8) -> u32;
        fn _mm_crc32_u16(crc: u32, v: u16) -> u32;
        fn _mm_crc32_u32(crc: u32, v: u32) -> u32;
        fn _mm_cmpgt_epi64(a: __m128i, b: __m128i) -> __m128i;
    }
}
