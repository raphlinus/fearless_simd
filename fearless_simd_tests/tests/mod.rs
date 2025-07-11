// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Simd, SimdFloat};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

// Ensure that we can cast between generic native-width vectors
#[allow(dead_code)]
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}
