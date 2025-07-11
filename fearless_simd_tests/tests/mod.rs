// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

#[test]
fn saturate_float_to_int() {
    #[inline(always)]
    fn helper_inner<S: Simd>(simd: S) {
        assert_eq!(
            <[u32; 4]>::from(simd.cvt_u32_f32x4(simd.splat_f32x4(f32::INFINITY))),
            [u32::MAX; 4]
        );
        assert_eq!(
            <[u32; 4]>::from(simd.cvt_u32_f32x4(simd.splat_f32x4(-f32::INFINITY))),
            [0; 4]
        );
        assert_eq!(
            <[i32; 4]>::from(simd.cvt_i32_f32x4(simd.splat_f32x4(f32::INFINITY))),
            [i32::MAX; 4]
        );
        assert_eq!(
            <[i32; 4]>::from(simd.cvt_i32_f32x4(simd.splat_f32x4(-f32::INFINITY))),
            [i32::MIN; 4]
        );
    }

    simd_dispatch!(helper(level) = helper_inner);
    helper(Level::new());
}
