// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the AVX2 level.
//!
//! This is short for the "x86-64-v3 microarchitecture level".

use core::arch::x86_64::*;

use crate::{
    f32x4,
    macros::{delegate, impl_op, impl_simd_from_into},
    mask32x4,
    seal::Seal,
    Fallback, Simd, SimdFrom, SimdInto, WithSimd,
};

/// The level enum for x86_64 architectures.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    Avx2(Avx2),
    // TODO: Avx512 (either nightly or pending stabilization)
}

/// The SIMD token for the "avx2" level.
///
/// This is short for the "x86-64-v3 microarchitecture level". In this level, the
/// following target_features are enabled: "avx2", "bmi2", "f16c", "fma", "lzcnt".
#[derive(Clone, Copy, Debug)]
pub struct Avx2 {
    _private: (),
}

impl Level {
    pub fn new() -> Self {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("lzcnt")
        {
            Level::Avx2(Avx2 { _private: () })
        } else {
            Level::Fallback(Fallback::new())
        }
    }

    pub fn as_avx2(self) -> Option<Avx2> {
        if let Level::Avx2(avx2) = self {
            Some(avx2)
        } else {
            None
        }
    }

    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[target_feature(enable = "avx2,bmi2,f16c,fma,lzcnt")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_avx2<W: WithSimd>(f: W, avx2: Avx2) -> W::Output {
            f.with_simd(avx2)
        }
        match self {
            Level::Fallback(fallback) => f.with_simd(fallback),
            Level::Avx2(avx2) => unsafe { dispatch_avx2(f, avx2) },
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

    #[inline(always)]
    fn splat_f32x4(self, val: f32) -> f32x4<Self> {
        self._mm_set1_ps(val).simd_into(self)
    }

    impl_op!(add_f32x4(a: f32x4, b: f32x4) -> f32x4 = _mm_add_ps);
    impl_op!(mul_f32x4(a: f32x4, b: f32x4) -> f32x4 = _mm_mul_ps);
    impl_op!(mul_add_f32x4(a: f32x4, b: f32x4, c: f32x4) -> f32x4 = _mm_fmadd_ps);

    impl_op!(simd_gt_f32x4(a: f32x4, b: f32x4) -> mask32x4 = _mm_castps_si128(_mm_cmpgt_ps));
    impl_op!(select_f32x4(a: mask32x4, b: f32x4, c: f32x4) -> f32x4
        = _mm_blendv_ps(c, b, _mm_castsi128_ps(a)));
    impl_op!(sqrt_f32x4(a: f32x4) -> f32x4 = _mm_sqrt_ps);

    #[inline(always)]
    fn abs_f32x4(self, a: f32x4<Self>) -> f32x4<Self> {
        let sign_mask = self._mm_castsi128_ps(self._mm_set1_epi32(0x7fff_ffff));
        self._mm_and_ps(sign_mask, a.into()).simd_into(self)
    }

    #[inline(always)]
    fn copysign_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let sign_mask = self._mm_castsi128_ps(self._mm_set1_epi32(-0x8000_0000));
        self._mm_or_ps(
            self._mm_and_ps(sign_mask, b.into()),
            self._mm_andnot_ps(sign_mask, a.into()),
        )
        .simd_into(self)
    }
}

// These implementations are cut and pasted from pulp. If we want to support
// a level lower than avx2, then we'll want to split it up.

/// Safety-wrapped intrinsics from avx2 level.
impl Avx2 {
    delegate! { core::arch::x86_64:
        // from Sse
        fn _mm_add_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_add_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_sub_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_sub_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_mul_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_mul_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_div_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_div_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_sqrt_ss(a: __m128) -> __m128;
        fn _mm_sqrt_ps(a: __m128) -> __m128;
        fn _mm_rcp_ss(a: __m128) -> __m128;
        fn _mm_rcp_ps(a: __m128) -> __m128;
        fn _mm_rsqrt_ss(a: __m128) -> __m128;
        fn _mm_rsqrt_ps(a: __m128) -> __m128;
        fn _mm_min_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_min_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_max_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_max_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_and_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_or_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_xor_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpeq_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmplt_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmple_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpgt_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpge_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpneq_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpnlt_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpnle_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpngt_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpnge_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpord_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpunord_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpeq_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmplt_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpge_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpneq_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpnlt_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpnle_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpngt_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpnge_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpord_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_cmpunord_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_comieq_ss(a: __m128, b: __m128) -> i32;
        fn _mm_comilt_ss(a: __m128, b: __m128) -> i32;
        fn _mm_comile_ss(a: __m128, b: __m128) -> i32;
        fn _mm_comigt_ss(a: __m128, b: __m128) -> i32;
        fn _mm_comige_ss(a: __m128, b: __m128) -> i32;
        fn _mm_comineq_ss(a: __m128, b: __m128) -> i32;
        fn _mm_ucomieq_ss(a: __m128, b: __m128) -> i32;
        fn _mm_ucomilt_ss(a: __m128, b: __m128) -> i32;
        fn _mm_ucomile_ss(a: __m128, b: __m128) -> i32;
        fn _mm_ucomigt_ss(a: __m128, b: __m128) -> i32;
        fn _mm_ucomige_ss(a: __m128, b: __m128) -> i32;
        fn _mm_ucomineq_ss(a: __m128, b: __m128) -> i32;
        fn _mm_cvtss_si32(a: __m128) -> i32;
        fn _mm_cvt_ss2si(a: __m128) -> i32;
        fn _mm_cvttss_si32(a: __m128) -> i32;
        fn _mm_cvtt_ss2si(a: __m128) -> i32;
        fn _mm_cvtss_f32(a: __m128) -> f32;
        fn _mm_cvtsi32_ss(a: __m128, b: i32) -> __m128;
        fn _mm_cvt_si2ss(a: __m128, b: i32) -> __m128;
        fn _mm_set_ss(a: f32) -> __m128;
        fn _mm_set1_ps(a: f32) -> __m128;
        fn _mm_set_ps1(a: f32) -> __m128;
        fn _mm_set_ps(a: f32, b: f32, c: f32, d: f32) -> __m128;
        fn _mm_setr_ps(a: f32, b: f32, c: f32, d: f32) -> __m128;
        fn _mm_setzero_ps() -> __m128;
        fn _mm_shuffle_ps<const MASK: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm_unpackhi_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_unpacklo_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_movehl_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_movelh_ps(a: __m128, b: __m128) -> __m128;
        fn _mm_movemask_ps(a: __m128) -> i32;
        unsafe fn _mm_load_ss(p: *const f32) -> __m128;
        unsafe fn _mm_load1_ps(p: *const f32) -> __m128;
        unsafe fn _mm_load_ps1(p: *const f32) -> __m128;
        unsafe fn _mm_load_ps(p: *const f32) -> __m128;
        unsafe fn _mm_loadu_ps(p: *const f32) -> __m128;
        unsafe fn _mm_loadr_ps(p: *const f32) -> __m128;
        unsafe fn _mm_loadu_si64(mem_addr: *const u8) -> __m128i;
        unsafe fn _mm_store_ss(p: *mut f32, a: __m128);
        unsafe fn _mm_store1_ps(p: *mut f32, a: __m128);
        unsafe fn _mm_store_ps1(p: *mut f32, a: __m128);
        unsafe fn _mm_store_ps(p: *mut f32, a: __m128);
        unsafe fn _mm_storeu_ps(p: *mut f32, a: __m128);
        unsafe fn _mm_storer_ps(p: *mut f32, a: __m128);
        fn _mm_move_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_sfence();
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn _mm_prefetch<const STRATEGY: i32>(p: *const i8);
        fn _mm_undefined_ps() -> __m128;
        #[allow(non_snake_case)]
        fn _MM_TRANSPOSE4_PS(
            row0: &mut __m128,
            row1: &mut __m128,
            row2: &mut __m128,
            row3: &mut __m128,
        );
        unsafe fn _mm_stream_ps(mem_addr: *mut f32, a: __m128);

        // from Sse2
        fn _mm_pause();
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn _mm_clflush(p: *const u8);
        fn _mm_lfence();
        fn _mm_mfence();
        fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_add_epi64(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_adds_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_adds_epu16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_avg_epu8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_avg_epu16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mulhi_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mulhi_epu16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mullo_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mul_epu32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sad_epu8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sub_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sub_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_sub_epi64(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_slli_si128<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_bslli_si128<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_bsrli_si128<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_slli_epi16<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_sll_epi16(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_slli_epi32<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_sll_epi32(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_slli_epi64<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_sll_epi64(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_srai_epi16<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_sra_epi16(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_srai_epi32<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_sra_epi32(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_srli_si128<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_srli_epi16<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_srl_epi16(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_srli_epi32<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_srl_epi32(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_srli_epi64<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_srl_epi64(a: __m128i, count: __m128i) -> __m128i;
        fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_xor_si128(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpeq_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmplt_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmplt_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmplt_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cvtepi32_pd(a: __m128i) -> __m128d;
        fn _mm_cvtsi32_sd(a: __m128d, b: i32) -> __m128d;
        fn _mm_cvtepi32_ps(a: __m128i) -> __m128;
        fn _mm_cvtps_epi32(a: __m128) -> __m128i;
        fn _mm_cvtsi32_si128(a: i32) -> __m128i;
        fn _mm_cvtsi128_si32(a: __m128i) -> i32;
        fn _mm_set_epi64x(e1: i64, e0: i64) -> __m128i;
        fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i;
        fn _mm_set_epi16(
            e7: i16,
            e6: i16,
            e5: i16,
            e4: i16,
            e3: i16,
            e2: i16,
            e1: i16,
            e0: i16,
        ) -> __m128i;
        fn _mm_set_epi8(
            e15: i8,
            e14: i8,
            e13: i8,
            e12: i8,
            e11: i8,
            e10: i8,
            e9: i8,
            e8: i8,
            e7: i8,
            e6: i8,
            e5: i8,
            e4: i8,
            e3: i8,
            e2: i8,
            e1: i8,
            e0: i8,
        ) -> __m128i;
        fn _mm_set1_epi64x(a: i64) -> __m128i;
        fn _mm_set1_epi32(a: i32) -> __m128i;
        fn _mm_set1_epi16(a: i16) -> __m128i;
        fn _mm_set1_epi8(a: i8) -> __m128i;
        fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i;
        fn _mm_setr_epi16(
            e7: i16,
            e6: i16,
            e5: i16,
            e4: i16,
            e3: i16,
            e2: i16,
            e1: i16,
            e0: i16,
        ) -> __m128i;
        fn _mm_setr_epi8(
            e15: i8,
            e14: i8,
            e13: i8,
            e12: i8,
            e11: i8,
            e10: i8,
            e9: i8,
            e8: i8,
            e7: i8,
            e6: i8,
            e5: i8,
            e4: i8,
            e3: i8,
            e2: i8,
            e1: i8,
            e0: i8,
        ) -> __m128i;
        fn _mm_setzero_si128() -> __m128i;
        unsafe fn _mm_loadl_epi64(mem_addr: *const __m128i) -> __m128i;
        unsafe fn _mm_load_si128(mem_addr: *const __m128i) -> __m128i;
        unsafe fn _mm_loadu_si128(mem_addr: *const __m128i) -> __m128i;
        unsafe fn _mm_maskmoveu_si128(a: __m128i, mask: __m128i, mem_addr: *mut i8);
        unsafe fn _mm_store_si128(mem_addr: *mut __m128i, a: __m128i);
        unsafe fn _mm_storeu_si128(mem_addr: *mut __m128i, a: __m128i);
        unsafe fn _mm_storel_epi64(mem_addr: *mut __m128i, a: __m128i);
        unsafe fn _mm_stream_si128(mem_addr: *mut __m128i, a: __m128i);
        unsafe fn _mm_stream_si32(mem_addr: *mut i32, a: i32);
        fn _mm_move_epi64(a: __m128i) -> __m128i;
        fn _mm_packs_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_packs_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_packus_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_extract_epi16<const IMM8: i32>(a: __m128i) -> i32;
        fn _mm_insert_epi16<const IMM8: i32>(a: __m128i, i: i32) -> __m128i;
        fn _mm_movemask_epi8(a: __m128i) -> i32;
        fn _mm_shuffle_epi32<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_shufflehi_epi16<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_shufflelo_epi16<const IMM8: i32>(a: __m128i) -> __m128i;
        fn _mm_unpackhi_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpacklo_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_div_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_div_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_max_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_max_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_min_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_min_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_mul_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_mul_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_sqrt_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_sqrt_pd(a: __m128d) -> __m128d;
        fn _mm_sub_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_sub_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_and_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_andnot_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_or_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_xor_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpeq_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmplt_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmple_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpgt_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpge_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpord_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpunord_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpneq_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpnlt_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpnle_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpngt_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpnge_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpeq_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmplt_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmple_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpgt_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpge_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpord_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpunord_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpneq_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpnlt_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpnle_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpngt_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmpnge_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_comieq_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_comilt_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_comile_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_comigt_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_comige_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_comineq_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_ucomieq_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_ucomilt_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_ucomile_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_ucomigt_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_ucomige_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_ucomineq_sd(a: __m128d, b: __m128d) -> i32;
        fn _mm_cvtpd_ps(a: __m128d) -> __m128;
        fn _mm_cvtps_pd(a: __m128) -> __m128d;
        fn _mm_cvtpd_epi32(a: __m128d) -> __m128i;
        fn _mm_cvtsd_si32(a: __m128d) -> i32;
        fn _mm_cvtsd_ss(a: __m128, b: __m128d) -> __m128;
        fn _mm_cvtsd_f64(a: __m128d) -> f64;
        fn _mm_cvtss_sd(a: __m128d, b: __m128) -> __m128d;
        fn _mm_cvttpd_epi32(a: __m128d) -> __m128i;
        fn _mm_cvttsd_si32(a: __m128d) -> i32;
        fn _mm_cvttps_epi32(a: __m128) -> __m128i;
        fn _mm_set_sd(a: f64) -> __m128d;
        fn _mm_set1_pd(a: f64) -> __m128d;
        fn _mm_set_pd1(a: f64) -> __m128d;
        fn _mm_set_pd(a: f64, b: f64) -> __m128d;
        fn _mm_setr_pd(a: f64, b: f64) -> __m128d;
        fn _mm_setzero_pd() -> __m128d;
        fn _mm_movemask_pd(a: __m128d) -> i32;
        unsafe fn _mm_load_pd(mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_load_sd(mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_loadh_pd(a: __m128d, mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_loadl_pd(a: __m128d, mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_stream_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_store_sd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_store_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_storeu_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_store1_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_store_pd1(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_storer_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_storeh_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_storel_pd(mem_addr: *mut f64, a: __m128d);
        unsafe fn _mm_load1_pd(mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_load_pd1(mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_loadr_pd(mem_addr: *const f64) -> __m128d;
        unsafe fn _mm_loadu_pd(mem_addr: *const f64) -> __m128d;
        fn _mm_shuffle_pd<const MASK: i32>(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_move_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_castpd_ps(a: __m128d) -> __m128;
        fn _mm_castpd_si128(a: __m128d) -> __m128i;
        fn _mm_castps_pd(a: __m128) -> __m128d;
        fn _mm_castps_si128(a: __m128) -> __m128i;
        fn _mm_castsi128_pd(a: __m128i) -> __m128d;
        fn _mm_castsi128_ps(a: __m128i) -> __m128;
        fn _mm_undefined_pd() -> __m128d;
        fn _mm_undefined_si128() -> __m128i;
        fn _mm_unpackhi_pd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_unpacklo_pd(a: __m128d, b: __m128d) -> __m128d;

        // from Sse3
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

        // from Ssse3
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

        // from Sse4_1
        fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i;
        fn _mm_blend_epi16<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) -> __m128d;
        fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) -> __m128;
        fn _mm_blend_pd<const IMM2: i32>(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_blend_ps<const IMM4: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm_extract_ps<const IMM8: i32>(a: __m128) -> i32;
        fn _mm_extract_epi8<const IMM8: i32>(a: __m128i) -> i32;
        fn _mm_extract_epi32<const IMM8: i32>(a: __m128i) -> i32;
        fn _mm_insert_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm_insert_epi8<const IMM8: i32>(a: __m128i, i: i32) -> __m128i;
        fn _mm_insert_epi32<const IMM8: i32>(a: __m128i, i: i32) -> __m128i;
        fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_max_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_min_epu16(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_min_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_min_epu32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_packus_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_cvtepi8_epi16(a: __m128i) -> __m128i;
        fn _mm_cvtepi8_epi32(a: __m128i) -> __m128i;
        fn _mm_cvtepi8_epi64(a: __m128i) -> __m128i;
        fn _mm_cvtepi16_epi32(a: __m128i) -> __m128i;
        fn _mm_cvtepi16_epi64(a: __m128i) -> __m128i;
        fn _mm_cvtepi32_epi64(a: __m128i) -> __m128i;
        fn _mm_cvtepu8_epi16(a: __m128i) -> __m128i;
        fn _mm_cvtepu8_epi32(a: __m128i) -> __m128i;
        fn _mm_cvtepu8_epi64(a: __m128i) -> __m128i;
        fn _mm_cvtepu16_epi32(a: __m128i) -> __m128i;
        fn _mm_cvtepu16_epi64(a: __m128i) -> __m128i;
        fn _mm_cvtepu32_epi64(a: __m128i) -> __m128i;
        fn _mm_dp_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_dp_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm_floor_pd(a: __m128d) -> __m128d;
        fn _mm_floor_ps(a: __m128) -> __m128;
        fn _mm_floor_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_floor_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_ceil_pd(a: __m128d) -> __m128d;
        fn _mm_ceil_ps(a: __m128) -> __m128;
        fn _mm_ceil_sd(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_ceil_ss(a: __m128, b: __m128) -> __m128;
        fn _mm_round_pd<const ROUNDING: i32>(a: __m128d) -> __m128d;
        fn _mm_round_ps<const ROUNDING: i32>(a: __m128) -> __m128;
        fn _mm_round_sd<const ROUNDING: i32>(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_round_ss<const ROUNDING: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm_minpos_epu16(a: __m128i) -> __m128i;
        fn _mm_mul_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mullo_epi32(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_mpsadbw_epu8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i;
        fn _mm_testz_si128(a: __m128i, mask: __m128i) -> i32;
        fn _mm_testc_si128(a: __m128i, mask: __m128i) -> i32;
        fn _mm_testnzc_si128(a: __m128i, mask: __m128i) -> i32;
        fn _mm_test_all_zeros(a: __m128i, mask: __m128i) -> i32;
        fn _mm_test_all_ones(a: __m128i) -> i32;
        fn _mm_test_mix_ones_zeros(a: __m128i, mask: __m128i) -> i32;

        // from Sse4_2
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

        // from Avx
        fn _mm256_add_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_add_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_and_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_and_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_or_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_or_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_shuffle_pd<const MASK: i32>(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_shuffle_ps<const MASK: i32>(a: __m256, b: __m256) -> __m256;
        fn _mm256_andnot_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_andnot_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_max_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_max_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_min_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_min_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_mul_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_mul_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_addsub_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_addsub_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_sub_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_sub_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_div_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_div_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_round_pd<const ROUNDING: i32>(a: __m256d) -> __m256d;
        fn _mm256_ceil_pd(a: __m256d) -> __m256d;
        fn _mm256_floor_pd(a: __m256d) -> __m256d;
        fn _mm256_round_ps<const ROUNDING: i32>(a: __m256) -> __m256;
        fn _mm256_ceil_ps(a: __m256) -> __m256;
        fn _mm256_floor_ps(a: __m256) -> __m256;
        fn _mm256_sqrt_ps(a: __m256) -> __m256;
        fn _mm256_sqrt_pd(a: __m256d) -> __m256d;
        fn _mm256_blend_pd<const IMM4: i32>(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_blend_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256;
        fn _mm256_blendv_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
        fn _mm256_blendv_ps(a: __m256, b: __m256, c: __m256) -> __m256;
        fn _mm256_dp_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256;
        fn _mm256_hadd_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_hadd_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_hsub_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_hsub_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_xor_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_xor_ps(a: __m256, b: __m256) -> __m256;
        fn _mm_cmp_pd<const IMM5: i32>(a: __m128d, b: __m128d) -> __m128d;
        fn _mm256_cmp_pd<const IMM5: i32>(a: __m256d, b: __m256d) -> __m256d;
        fn _mm_cmp_ps<const IMM5: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm256_cmp_ps<const IMM5: i32>(a: __m256, b: __m256) -> __m256;
        fn _mm_cmp_sd<const IMM5: i32>(a: __m128d, b: __m128d) -> __m128d;
        fn _mm_cmp_ss<const IMM5: i32>(a: __m128, b: __m128) -> __m128;
        fn _mm256_cvtepi32_pd(a: __m128i) -> __m256d;
        fn _mm256_cvtepi32_ps(a: __m256i) -> __m256;
        fn _mm256_cvtpd_ps(a: __m256d) -> __m128;
        fn _mm256_cvtps_epi32(a: __m256) -> __m256i;
        fn _mm256_cvtps_pd(a: __m128) -> __m256d;
        fn _mm256_cvttpd_epi32(a: __m256d) -> __m128i;
        fn _mm256_cvtpd_epi32(a: __m256d) -> __m128i;
        fn _mm256_cvttps_epi32(a: __m256) -> __m256i;
        fn _mm256_extractf128_ps<const IMM1: i32>(a: __m256) -> __m128;
        fn _mm256_extractf128_pd<const IMM1: i32>(a: __m256d) -> __m128d;
        fn _mm256_extractf128_si256<const IMM1: i32>(a: __m256i) -> __m128i;
        fn _mm256_zeroall();
        fn _mm256_zeroupper();
        fn _mm256_permutevar_ps(a: __m256, b: __m256i) -> __m256;
        fn _mm_permutevar_ps(a: __m128, b: __m128i) -> __m128;
        fn _mm256_permute_ps<const IMM8: i32>(a: __m256) -> __m256;
        fn _mm_permute_ps<const IMM8: i32>(a: __m128) -> __m128;
        fn _mm256_permutevar_pd(a: __m256d, b: __m256i) -> __m256d;
        fn _mm_permutevar_pd(a: __m128d, b: __m128i) -> __m128d;
        fn _mm256_permute_pd<const IMM4: i32>(a: __m256d) -> __m256d;
        fn _mm_permute_pd<const IMM2: i32>(a: __m128d) -> __m128d;
        fn _mm256_permute2f128_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256;
        fn _mm256_permute2f128_pd<const IMM8: i32>(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_permute2f128_si256<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_broadcast_ss(f: &f32) -> __m256;
        fn _mm_broadcast_ss(f: &f32) -> __m128;
        fn _mm256_broadcast_sd(f: &f64) -> __m256d;
        fn _mm256_broadcast_ps(a: &__m128) -> __m256;
        fn _mm256_broadcast_pd(a: &__m128d) -> __m256d;
        fn _mm256_insertf128_ps<const IMM1: i32>(a: __m256, b: __m128) -> __m256;
        fn _mm256_insertf128_pd<const IMM1: i32>(a: __m256d, b: __m128d) -> __m256d;
        fn _mm256_insertf128_si256<const IMM1: i32>(a: __m256i, b: __m128i) -> __m256i;
        fn _mm256_insert_epi8<const INDEX: i32>(a: __m256i, i: i8) -> __m256i;
        fn _mm256_insert_epi16<const INDEX: i32>(a: __m256i, i: i16) -> __m256i;
        fn _mm256_insert_epi32<const INDEX: i32>(a: __m256i, i: i32) -> __m256i;
        unsafe fn _mm256_load_pd(mem_addr: *const f64) -> __m256d;
        unsafe fn _mm256_store_pd(mem_addr: *mut f64, a: __m256d);
        unsafe fn _mm256_load_ps(mem_addr: *const f32) -> __m256;
        unsafe fn _mm256_store_ps(mem_addr: *mut f32, a: __m256);
        unsafe fn _mm256_loadu_pd(mem_addr: *const f64) -> __m256d;
        unsafe fn _mm256_storeu_pd(mem_addr: *mut f64, a: __m256d);
        unsafe fn _mm256_loadu_ps(mem_addr: *const f32) -> __m256;
        unsafe fn _mm256_storeu_ps(mem_addr: *mut f32, a: __m256);
        unsafe fn _mm256_load_si256(mem_addr: *const __m256i) -> __m256i;
        unsafe fn _mm256_store_si256(mem_addr: *mut __m256i, a: __m256i);
        unsafe fn _mm256_loadu_si256(mem_addr: *const __m256i) -> __m256i;
        unsafe fn _mm256_storeu_si256(mem_addr: *mut __m256i, a: __m256i);
        unsafe fn _mm256_maskload_pd(mem_addr: *const f64, mask: __m256i) -> __m256d;
        unsafe fn _mm256_maskstore_pd(mem_addr: *mut f64, mask: __m256i, a: __m256d);
        unsafe fn _mm_maskload_pd(mem_addr: *const f64, mask: __m128i) -> __m128d;
        unsafe fn _mm_maskstore_pd(mem_addr: *mut f64, mask: __m128i, a: __m128d);
        unsafe fn _mm256_maskload_ps(mem_addr: *const f32, mask: __m256i) -> __m256;
        unsafe fn _mm256_maskstore_ps(mem_addr: *mut f32, mask: __m256i, a: __m256);
        unsafe fn _mm_maskload_ps(mem_addr: *const f32, mask: __m128i) -> __m128;
        unsafe fn _mm_maskstore_ps(mem_addr: *mut f32, mask: __m128i, a: __m128);
        fn _mm256_movehdup_ps(a: __m256) -> __m256;
        fn _mm256_moveldup_ps(a: __m256) -> __m256;
        fn _mm256_movedup_pd(a: __m256d) -> __m256d;
        unsafe fn _mm256_lddqu_si256(mem_addr: *const __m256i) -> __m256i;
        unsafe fn _mm256_stream_si256(mem_addr: *mut __m256i, a: __m256i);
        unsafe fn _mm256_stream_pd(mem_addr: *mut f64, a: __m256d);
        unsafe fn _mm256_stream_ps(mem_addr: *mut f32, a: __m256);
        fn _mm256_rcp_ps(a: __m256) -> __m256;
        fn _mm256_rsqrt_ps(a: __m256) -> __m256;
        fn _mm256_unpackhi_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_unpackhi_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_unpacklo_pd(a: __m256d, b: __m256d) -> __m256d;
        fn _mm256_unpacklo_ps(a: __m256, b: __m256) -> __m256;
        fn _mm256_testz_si256(a: __m256i, b: __m256i) -> i32;
        fn _mm256_testc_si256(a: __m256i, b: __m256i) -> i32;
        fn _mm256_testnzc_si256(a: __m256i, b: __m256i) -> i32;
        fn _mm256_testz_pd(a: __m256d, b: __m256d) -> i32;
        fn _mm256_testc_pd(a: __m256d, b: __m256d) -> i32;
        fn _mm256_testnzc_pd(a: __m256d, b: __m256d) -> i32;
        fn _mm_testz_pd(a: __m128d, b: __m128d) -> i32;
        fn _mm_testc_pd(a: __m128d, b: __m128d) -> i32;
        fn _mm_testnzc_pd(a: __m128d, b: __m128d) -> i32;
        fn _mm256_testz_ps(a: __m256, b: __m256) -> i32;
        fn _mm256_testc_ps(a: __m256, b: __m256) -> i32;
        fn _mm256_testnzc_ps(a: __m256, b: __m256) -> i32;
        fn _mm_testz_ps(a: __m128, b: __m128) -> i32;
        fn _mm_testc_ps(a: __m128, b: __m128) -> i32;
        fn _mm_testnzc_ps(a: __m128, b: __m128) -> i32;
        fn _mm256_movemask_pd(a: __m256d) -> i32;
        fn _mm256_movemask_ps(a: __m256) -> i32;
        fn _mm256_setzero_pd() -> __m256d;
        fn _mm256_setzero_ps() -> __m256;
        fn _mm256_setzero_si256() -> __m256i;
        fn _mm256_set_pd(a: f64, b: f64, c: f64, d: f64) -> __m256d;
        fn _mm256_set_ps(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> __m256;
        fn _mm256_set_epi8(
            e00: i8,
            e01: i8,
            e02: i8,
            e03: i8,
            e04: i8,
            e05: i8,
            e06: i8,
            e07: i8,
            e08: i8,
            e09: i8,
            e10: i8,
            e11: i8,
            e12: i8,
            e13: i8,
            e14: i8,
            e15: i8,
            e16: i8,
            e17: i8,
            e18: i8,
            e19: i8,
            e20: i8,
            e21: i8,
            e22: i8,
            e23: i8,
            e24: i8,
            e25: i8,
            e26: i8,
            e27: i8,
            e28: i8,
            e29: i8,
            e30: i8,
            e31: i8,
        ) -> __m256i;
        fn _mm256_set_epi16(
            e00: i16,
            e01: i16,
            e02: i16,
            e03: i16,
            e04: i16,
            e05: i16,
            e06: i16,
            e07: i16,
            e08: i16,
            e09: i16,
            e10: i16,
            e11: i16,
            e12: i16,
            e13: i16,
            e14: i16,
            e15: i16,
        ) -> __m256i;
        fn _mm256_set_epi32(
            e0: i32,
            e1: i32,
            e2: i32,
            e3: i32,
            e4: i32,
            e5: i32,
            e6: i32,
            e7: i32,
        ) -> __m256i;
        fn _mm256_set_epi64x(a: i64, b: i64, c: i64, d: i64) -> __m256i;
        fn _mm256_setr_pd(a: f64, b: f64, c: f64, d: f64) -> __m256d;
        fn _mm256_setr_ps(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32)
        -> __m256;
        fn _mm256_setr_epi8(
            e00: i8,
            e01: i8,
            e02: i8,
            e03: i8,
            e04: i8,
            e05: i8,
            e06: i8,
            e07: i8,
            e08: i8,
            e09: i8,
            e10: i8,
            e11: i8,
            e12: i8,
            e13: i8,
            e14: i8,
            e15: i8,
            e16: i8,
            e17: i8,
            e18: i8,
            e19: i8,
            e20: i8,
            e21: i8,
            e22: i8,
            e23: i8,
            e24: i8,
            e25: i8,
            e26: i8,
            e27: i8,
            e28: i8,
            e29: i8,
            e30: i8,
            e31: i8,
        ) -> __m256i;
        fn _mm256_setr_epi16(
            e00: i16,
            e01: i16,
            e02: i16,
            e03: i16,
            e04: i16,
            e05: i16,
            e06: i16,
            e07: i16,
            e08: i16,
            e09: i16,
            e10: i16,
            e11: i16,
            e12: i16,
            e13: i16,
            e14: i16,
            e15: i16,
        ) -> __m256i;
        fn _mm256_setr_epi32(
            e0: i32,
            e1: i32,
            e2: i32,
            e3: i32,
            e4: i32,
            e5: i32,
            e6: i32,
            e7: i32,
        ) -> __m256i;
        fn _mm256_setr_epi64x(a: i64, b: i64, c: i64, d: i64) -> __m256i;
        fn _mm256_set1_pd(a: f64) -> __m256d;
        fn _mm256_set1_ps(a: f32) -> __m256;
        fn _mm256_set1_epi8(a: i8) -> __m256i;
        fn _mm256_set1_epi16(a: i16) -> __m256i;
        fn _mm256_set1_epi32(a: i32) -> __m256i;
        fn _mm256_set1_epi64x(a: i64) -> __m256i;
        fn _mm256_castpd_ps(a: __m256d) -> __m256;
        fn _mm256_castps_pd(a: __m256) -> __m256d;
        fn _mm256_castps_si256(a: __m256) -> __m256i;
        fn _mm256_castsi256_ps(a: __m256i) -> __m256;
        fn _mm256_castpd_si256(a: __m256d) -> __m256i;
        fn _mm256_castsi256_pd(a: __m256i) -> __m256d;
        fn _mm256_castps256_ps128(a: __m256) -> __m128;
        fn _mm256_castpd256_pd128(a: __m256d) -> __m128d;
        fn _mm256_castsi256_si128(a: __m256i) -> __m128i;
        fn _mm256_castps128_ps256(a: __m128) -> __m256;
        fn _mm256_castpd128_pd256(a: __m128d) -> __m256d;
        fn _mm256_castsi128_si256(a: __m128i) -> __m256i;
        fn _mm256_zextps128_ps256(a: __m128) -> __m256;
        fn _mm256_zextsi128_si256(a: __m128i) -> __m256i;
        fn _mm256_zextpd128_pd256(a: __m128d) -> __m256d;
        fn _mm256_undefined_ps() -> __m256;
        fn _mm256_undefined_pd() -> __m256d;
        fn _mm256_undefined_si256() -> __m256i;
        fn _mm256_set_m128(hi: __m128, lo: __m128) -> __m256;
        fn _mm256_set_m128d(hi: __m128d, lo: __m128d) -> __m256d;
        fn _mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i;
        fn _mm256_setr_m128(lo: __m128, hi: __m128) -> __m256;
        fn _mm256_setr_m128d(lo: __m128d, hi: __m128d) -> __m256d;
        fn _mm256_setr_m128i(lo: __m128i, hi: __m128i) -> __m256i;
        unsafe fn _mm256_loadu2_m128(hiaddr: *const f32, loaddr: *const f32) -> __m256;
        unsafe fn _mm256_loadu2_m128d(hiaddr: *const f64, loaddr: *const f64) -> __m256d;
        unsafe fn _mm256_loadu2_m128i(hiaddr: *const __m128i, loaddr: *const __m128i) -> __m256i;
        unsafe fn _mm256_storeu2_m128(hiaddr: *mut f32, loaddr: *mut f32, a: __m256);
        unsafe fn _mm256_storeu2_m128d(hiaddr: *mut f64, loaddr: *mut f64, a: __m256d);
        unsafe fn _mm256_storeu2_m128i(hiaddr: *mut __m128i, loaddr: *mut __m128i, a: __m256i);
        fn _mm256_cvtss_f32(a: __m256) -> f32;

        // from Avx2
        fn _mm256_abs_epi32(a: __m256i) -> __m256i;
        fn _mm256_abs_epi16(a: __m256i) -> __m256i;
        fn _mm256_abs_epi8(a: __m256i) -> __m256i;
        fn _mm256_add_epi64(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_add_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_add_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_add_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_adds_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_adds_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_adds_epu8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_adds_epu16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_alignr_epi8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_andnot_si256(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_avg_epu16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_avg_epu8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm_blend_epi32<const IMM4: i32>(a: __m128i, b: __m128i) -> __m128i;
        fn _mm256_blend_epi32<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_blend_epi16<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_blendv_epi8(a: __m256i, b: __m256i, mask: __m256i) -> __m256i;
        fn _mm_broadcastb_epi8(a: __m128i) -> __m128i;
        fn _mm256_broadcastb_epi8(a: __m128i) -> __m256i;
        fn _mm_broadcastd_epi32(a: __m128i) -> __m128i;
        fn _mm256_broadcastd_epi32(a: __m128i) -> __m256i;
        fn _mm_broadcastq_epi64(a: __m128i) -> __m128i;
        fn _mm256_broadcastq_epi64(a: __m128i) -> __m256i;
        fn _mm_broadcastsd_pd(a: __m128d) -> __m128d;
        fn _mm256_broadcastsd_pd(a: __m128d) -> __m256d;
        fn _mm256_broadcastsi128_si256(a: __m128i) -> __m256i;
        fn _mm_broadcastss_ps(a: __m128) -> __m128;
        fn _mm256_broadcastss_ps(a: __m128) -> __m256;
        fn _mm_broadcastw_epi16(a: __m128i) -> __m128i;
        fn _mm256_broadcastw_epi16(a: __m128i) -> __m256i;
        fn _mm256_cmpeq_epi64(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpeq_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpeq_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpeq_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpgt_epi64(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpgt_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpgt_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cmpgt_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_cvtepi16_epi32(a: __m128i) -> __m256i;
        fn _mm256_cvtepi16_epi64(a: __m128i) -> __m256i;
        fn _mm256_cvtepi32_epi64(a: __m128i) -> __m256i;
        fn _mm256_cvtepi8_epi16(a: __m128i) -> __m256i;
        fn _mm256_cvtepi8_epi32(a: __m128i) -> __m256i;
        fn _mm256_cvtepi8_epi64(a: __m128i) -> __m256i;
        fn _mm256_cvtepu16_epi32(a: __m128i) -> __m256i;
        fn _mm256_cvtepu16_epi64(a: __m128i) -> __m256i;
        fn _mm256_cvtepu32_epi64(a: __m128i) -> __m256i;
        fn _mm256_cvtepu8_epi16(a: __m128i) -> __m256i;
        fn _mm256_cvtepu8_epi32(a: __m128i) -> __m256i;
        fn _mm256_cvtepu8_epi64(a: __m128i) -> __m256i;
        fn _mm256_extracti128_si256<const IMM1: i32>(a: __m256i) -> __m128i;
        fn _mm256_hadd_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_hadd_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_hadds_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_hsub_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_hsub_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_hsubs_epi16(a: __m256i, b: __m256i) -> __m256i;
        unsafe fn _mm_i32gather_epi32<const SCALE: i32>(
            slice: *const i32,
            offsets: __m128i,
        ) -> __m128i;
        unsafe fn _mm_mask_i32gather_epi32<const SCALE: i32>(
            src: __m128i,
            slice: *const i32,
            offsets: __m128i,
            mask: __m128i,
        ) -> __m128i;
        unsafe fn _mm256_i32gather_epi32<const SCALE: i32>(
            slice: *const i32,
            offsets: __m256i,
        ) -> __m256i;
        unsafe fn _mm256_mask_i32gather_epi32<const SCALE: i32>(
            src: __m256i,
            slice: *const i32,
            offsets: __m256i,
            mask: __m256i,
        ) -> __m256i;
        unsafe fn _mm_i32gather_ps<const SCALE: i32>(slice: *const f32, offsets: __m128i)
        -> __m128;
        unsafe fn _mm_mask_i32gather_ps<const SCALE: i32>(
            src: __m128,
            slice: *const f32,
            offsets: __m128i,
            mask: __m128,
        ) -> __m128;
        unsafe fn _mm256_i32gather_ps<const SCALE: i32>(
            slice: *const f32,
            offsets: __m256i,
        ) -> __m256;
        unsafe fn _mm256_mask_i32gather_ps<const SCALE: i32>(
            src: __m256,
            slice: *const f32,
            offsets: __m256i,
            mask: __m256,
        ) -> __m256;
        unsafe fn _mm_i32gather_epi64<const SCALE: i32>(
            slice: *const i64,
            offsets: __m128i,
        ) -> __m128i;
        unsafe fn _mm_mask_i32gather_epi64<const SCALE: i32>(
            src: __m128i,
            slice: *const i64,
            offsets: __m128i,
            mask: __m128i,
        ) -> __m128i;
        unsafe fn _mm256_i32gather_epi64<const SCALE: i32>(
            slice: *const i64,
            offsets: __m128i,
        ) -> __m256i;
        unsafe fn _mm256_mask_i32gather_epi64<const SCALE: i32>(
            src: __m256i,
            slice: *const i64,
            offsets: __m128i,
            mask: __m256i,
        ) -> __m256i;
        unsafe fn _mm_i32gather_pd<const SCALE: i32>(
            slice: *const f64,
            offsets: __m128i,
        ) -> __m128d;
        unsafe fn _mm_mask_i32gather_pd<const SCALE: i32>(
            src: __m128d,
            slice: *const f64,
            offsets: __m128i,
            mask: __m128d,
        ) -> __m128d;
        unsafe fn _mm256_i32gather_pd<const SCALE: i32>(
            slice: *const f64,
            offsets: __m128i,
        ) -> __m256d;
        unsafe fn _mm256_mask_i32gather_pd<const SCALE: i32>(
            src: __m256d,
            slice: *const f64,
            offsets: __m128i,
            mask: __m256d,
        ) -> __m256d;
        unsafe fn _mm_i64gather_epi32<const SCALE: i32>(
            slice: *const i32,
            offsets: __m128i,
        ) -> __m128i;
        unsafe fn _mm_mask_i64gather_epi32<const SCALE: i32>(
            src: __m128i,
            slice: *const i32,
            offsets: __m128i,
            mask: __m128i,
        ) -> __m128i;
        unsafe fn _mm256_i64gather_epi32<const SCALE: i32>(
            slice: *const i32,
            offsets: __m256i,
        ) -> __m128i;
        unsafe fn _mm256_mask_i64gather_epi32<const SCALE: i32>(
            src: __m128i,
            slice: *const i32,
            offsets: __m256i,
            mask: __m128i,
        ) -> __m128i;
        unsafe fn _mm_i64gather_ps<const SCALE: i32>(slice: *const f32, offsets: __m128i)
        -> __m128;
        unsafe fn _mm_mask_i64gather_ps<const SCALE: i32>(
            src: __m128,
            slice: *const f32,
            offsets: __m128i,
            mask: __m128,
        ) -> __m128;
        unsafe fn _mm256_i64gather_ps<const SCALE: i32>(
            slice: *const f32,
            offsets: __m256i,
        ) -> __m128;
        unsafe fn _mm256_mask_i64gather_ps<const SCALE: i32>(
            src: __m128,
            slice: *const f32,
            offsets: __m256i,
            mask: __m128,
        ) -> __m128;
        unsafe fn _mm_i64gather_epi64<const SCALE: i32>(
            slice: *const i64,
            offsets: __m128i,
        ) -> __m128i;
        unsafe fn _mm_mask_i64gather_epi64<const SCALE: i32>(
            src: __m128i,
            slice: *const i64,
            offsets: __m128i,
            mask: __m128i,
        ) -> __m128i;
        unsafe fn _mm256_i64gather_epi64<const SCALE: i32>(
            slice: *const i64,
            offsets: __m256i,
        ) -> __m256i;
        unsafe fn _mm256_mask_i64gather_epi64<const SCALE: i32>(
            src: __m256i,
            slice: *const i64,
            offsets: __m256i,
            mask: __m256i,
        ) -> __m256i;
        unsafe fn _mm_i64gather_pd<const SCALE: i32>(
            slice: *const f64,
            offsets: __m128i,
        ) -> __m128d;
        unsafe fn _mm_mask_i64gather_pd<const SCALE: i32>(
            src: __m128d,
            slice: *const f64,
            offsets: __m128i,
            mask: __m128d,
        ) -> __m128d;
        unsafe fn _mm256_i64gather_pd<const SCALE: i32>(
            slice: *const f64,
            offsets: __m256i,
        ) -> __m256d;
        unsafe fn _mm256_mask_i64gather_pd<const SCALE: i32>(
            src: __m256d,
            slice: *const f64,
            offsets: __m256i,
            mask: __m256d,
        ) -> __m256d;
        fn _mm256_inserti128_si256<const IMM1: i32>(a: __m256i, b: __m128i) -> __m256i;
        fn _mm256_madd_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_maddubs_epi16(a: __m256i, b: __m256i) -> __m256i;
        unsafe fn _mm_maskload_epi32(mem_addr: *const i32, mask: __m128i) -> __m128i;
        unsafe fn _mm256_maskload_epi32(mem_addr: *const i32, mask: __m256i) -> __m256i;
        unsafe fn _mm_maskload_epi64(mem_addr: *const i64, mask: __m128i) -> __m128i;
        unsafe fn _mm256_maskload_epi64(mem_addr: *const i64, mask: __m256i) -> __m256i;
        unsafe fn _mm_maskstore_epi32(mem_addr: *mut i32, mask: __m128i, a: __m128i);
        unsafe fn _mm256_maskstore_epi32(mem_addr: *mut i32, mask: __m256i, a: __m256i);
        unsafe fn _mm_maskstore_epi64(mem_addr: *mut i64, mask: __m128i, a: __m128i);
        unsafe fn _mm256_maskstore_epi64(mem_addr: *mut i64, mask: __m256i, a: __m256i);
        fn _mm256_max_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_max_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_max_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_max_epu16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_max_epu32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_max_epu8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_min_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_min_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_min_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_min_epu16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_min_epu32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_min_epu8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_movemask_epi8(a: __m256i) -> i32;
        fn _mm256_mpsadbw_epu8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mul_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mul_epu32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mulhi_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mulhi_epu16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mullo_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mullo_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_mulhrs_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_or_si256(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_packs_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_packs_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_packus_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_packus_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_permutevar8x32_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_permute4x64_epi64<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_permute2x128_si256<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_permute4x64_pd<const IMM8: i32>(a: __m256d) -> __m256d;
        fn _mm256_permutevar8x32_ps(a: __m256, idx: __m256i) -> __m256;
        fn _mm256_sad_epu8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_shuffle_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_shuffle_epi32<const MASK: i32>(a: __m256i) -> __m256i;
        fn _mm256_shufflehi_epi16<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_shufflelo_epi16<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_sign_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_sign_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_sign_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_sll_epi16(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_sll_epi32(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_sll_epi64(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_slli_epi16<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_slli_epi32<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_slli_epi64<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_slli_si256<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_bslli_epi128<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm_sllv_epi32(a: __m128i, count: __m128i) -> __m128i;
        fn _mm256_sllv_epi32(a: __m256i, count: __m256i) -> __m256i;
        fn _mm_sllv_epi64(a: __m128i, count: __m128i) -> __m128i;
        fn _mm256_sllv_epi64(a: __m256i, count: __m256i) -> __m256i;
        fn _mm256_sra_epi16(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_sra_epi32(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_srai_epi16<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_srai_epi32<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm_srav_epi32(a: __m128i, count: __m128i) -> __m128i;
        fn _mm256_srav_epi32(a: __m256i, count: __m256i) -> __m256i;
        fn _mm256_srli_si256<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_bsrli_epi128<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_srl_epi16(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_srl_epi32(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_srl_epi64(a: __m256i, count: __m128i) -> __m256i;
        fn _mm256_srli_epi16<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_srli_epi32<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm256_srli_epi64<const IMM8: i32>(a: __m256i) -> __m256i;
        fn _mm_srlv_epi32(a: __m128i, count: __m128i) -> __m128i;
        fn _mm256_srlv_epi32(a: __m256i, count: __m256i) -> __m256i;
        fn _mm_srlv_epi64(a: __m128i, count: __m128i) -> __m128i;
        fn _mm256_srlv_epi64(a: __m256i, count: __m256i) -> __m256i;
        fn _mm256_sub_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_sub_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_sub_epi64(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_sub_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_subs_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_subs_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_subs_epu16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_subs_epu8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpackhi_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpacklo_epi8(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpackhi_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpacklo_epi16(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpackhi_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpacklo_epi32(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpackhi_epi64(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_unpacklo_epi64(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i;
        fn _mm256_extract_epi8<const INDEX: i32>(a: __m256i) -> i32;
        fn _mm256_extract_epi16<const INDEX: i32>(a: __m256i) -> i32;
        fn _mm256_extract_epi32<const INDEX: i32>(a: __m256i) -> i32;
        fn _mm256_cvtsd_f64(a: __m256d) -> f64;
        fn _mm256_cvtsi256_si32(a: __m256i) -> i32;

        // from Fma
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
