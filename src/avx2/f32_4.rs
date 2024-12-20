// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f32x4`] SIMD values.

use core::arch::x86_64::*;

use crate::{
    f32x4,
    macros::{impl_binop, impl_cmp, impl_select, impl_simd_from_into, impl_ternary, impl_unaryop},
};

impl_simd_from_into!(f32x4, __m128);

impl_unaryop!("sse": sqrt(f32x4) = _mm_sqrt_ps);
impl_binop!("sse": add(f32x4) = _mm_add_ps);
impl_binop!("sse": sub(f32x4) = _mm_sub_ps);
impl_binop!("sse": mul(f32x4) = _mm_mul_ps);
impl_binop!("sse": div(f32x4) = _mm_div_ps);
// Note: Intel does not follow IEEE rules wrt signed zero and NaN
impl_binop!("sse": min(f32x4) = _mm_min_ps);
impl_binop!("sse": max(f32x4) = _mm_max_ps);
impl_ternary!("fma": mul_add(f32x4) = _mm_fmadd_ps);
impl_ternary!("fma": mul_sub(f32x4) = _mm_fnmadd_ps);
impl_cmp!("sse": simd_eq(f32x4) = _mm_cmpeq_ps);
impl_cmp!("sse": simd_le(f32x4) = _mm_cmple_ps);
impl_cmp!("sse": simd_lt(f32x4) = _mm_cmplt_ps);
impl_cmp!("sse": simd_gt(f32x4) = _mm_cmpgt_ps);
impl_cmp!("sse": simd_ge(f32x4) = _mm_cmpge_ps);
// only seem to be signed integer casts on Intel
impl_select!("sse4.1": (f32x4) = _mm_blendv_ps cba);

#[target_feature(enable = "sse")]
#[inline]
pub fn neg(value: f32x4) -> f32x4 {
    unsafe { _mm_sub_ps(_mm_set1_ps(0.0), value.into()).into() }
}

#[target_feature(enable = "sse")]
#[inline]
pub fn abs(value: f32x4) -> f32x4 {
    unsafe {
        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
        _mm_and_ps(sign_mask, value.into()).into()
    }
}

#[target_feature(enable = "sse")]
#[inline]
pub fn copysign(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(-0x8000_0000));
        _mm_or_ps(
            _mm_and_ps(sign_mask, b.into()),
            _mm_andnot_ps(sign_mask, a.into()),
        )
        .into()
    }
}

#[target_feature(enable = "sse")]
#[inline]
pub fn splat(value: f32) -> f32x4 {
    unsafe { _mm_set1_ps(value).into() }
}

#[target_feature(enable = "sse4.1")]
#[inline]
pub fn copy_lane<const LANE1: i32, const LANE2: i32>(a: f32x4, b: f32x4) -> f32x4 {
    unsafe {
        let lhs = a.into();
        let rhs = _mm_castps_si128(b.into());
        let aligned = match (LANE1, LANE2) {
            (0, 1) | (1, 2) | (2, 3) => _mm_bsrli_si128::<4>(rhs),
            (0, 2) | (1, 3) => _mm_bsrli_si128::<8>(rhs),
            (0, 3) => _mm_bsrli_si128::<12>(rhs),
            (3, 0) => _mm_bslli_si128::<12>(rhs),
            (2, 0) | (3, 1) => _mm_bslli_si128::<8>(rhs),
            (1, 0) | (2, 1) | (3, 2) => _mm_bslli_si128::<4>(rhs),
            _ => rhs,
        };
        let cast = _mm_castsi128_ps(aligned);
        let blended = match LANE1 {
            0 => _mm_blend_ps::<1>(lhs, cast),
            1 => _mm_blend_ps::<2>(lhs, cast),
            2 => _mm_blend_ps::<4>(lhs, cast),
            3 => _mm_blend_ps::<8>(lhs, cast),
            // TODO: can we make this a static assert?
            _ => panic!("invalid lane index for type"),
        };
        blended.into()
    }
}
