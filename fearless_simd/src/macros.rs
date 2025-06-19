// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

#[cfg(feature = "std")]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "neon")]
            #[inline]
            unsafe fn inner_neon(neon: $crate::aarch64::Neon $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( neon $( , $arg )* )
            }
            #[cfg(target_arch = "wasm32")]
            #[target_feature(enable = "simd128")]
            #[inline]
            unsafe fn inner_wasm_simd128(simd128: $crate::wasm32::WasmSimd128 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( simd128 $( , $arg )* )
            }
            match level {
                Level::Fallback(fb) => $inner(fb $( , $arg )* ),
                #[cfg(target_arch = "aarch64")]
                Level::Neon(neon) => unsafe { inner_neon (neon $( , $arg )* ) }
                #[cfg(target_arch = "wasm32")]
                Level::WasmSimd128(wasm) => unsafe { inner_wasm_simd128 (wasm $( , $arg )* ) }
            }
        }
    };
}

#[cfg(not(feature = "std"))]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[cfg(target_arch = "wasm32")]
            #[target_feature(enable = "simd128")]
            #[inline]
            unsafe fn inner_wasm_simd128(simd128: $crate::wasm32::WasmSimd128 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( simd128 $( , $arg )* )
            }
            match level {
                Level::Fallback(fb) => $inner(fb $( , $arg )* ),
                #[cfg(target_arch = "wasm32")]
                Level::WasmSimd128(wasm) => unsafe { inner_wasm_simd128 (wasm $( , $arg )* ) }
            }
        }
    };
}
