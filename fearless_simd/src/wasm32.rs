// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for WebAssembly SIMD capabilities.


use crate::WithSimd;
pub use crate::generated::WasmSimd128;

/// A `Level` enum, which is maybe not required for compile time determined WASM.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    WasmSimd128(WasmSimd128),
    // TODO: Fallback(Fallback),
}

impl Default for Level {
    fn default() -> Self {
        Self::new()
    }
}

impl Level {
    pub fn new() -> Self {
        #[cfg(target_feature = "simd128")]
        {
            unsafe { Level::WasmSimd128(WasmSimd128::new_unchecked()) }
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            // TODO: Return fallback level.
            panic!("WASM SIMD128 not available. Compile with target-feature=+simd128");
        }
    }

    #[inline]
    pub fn as_wasm_simd128(self) -> Option<WasmSimd128> {
        match self {
            Level::WasmSimd128(simd128) => Some(simd128),
        }
    }

    #[inline]
    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn dispatch_simd128<W: WithSimd>(f: W, simd128: WasmSimd128) -> W::Output {
            f.with_simd(simd128)
        }
        match self {
            Level::WasmSimd128(simd128) => dispatch_simd128(f, simd128),
        }
    }
}