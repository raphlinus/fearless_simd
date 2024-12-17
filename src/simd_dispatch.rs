// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A macro for writing dispatch stanzas.

/// Create a dispatch block.
///
/// The argument is a normal Rust function, though with some syntax limitations
/// (arguments must be single identifers, no generics, no where clauses). It
/// creates a module (of the same name as the function) with a separate function
/// for each level. It also creates a dispatch function that detects cpu level
/// and calls into the appropriate version.
///
/// While this is a modestly impressive declarative macro, it really needs to be
/// a proc macro, as it needs to support cfg-like conditional compilation. And
/// I'm sure there other issues.
#[macro_export]
macro_rules! simd_dispatch {
    ( #[ levels = ( $( $level:ident ),+ ) ] $vis:vis $f:ident ( $( $arg:tt )* ) $(-> $retty:ty )? $body:block) => {
        mod $f {
            $crate::simd_dispatch!(@cases [ $( $level )+ ] $vis $f { ( $( $arg )* ) $( -> $retty )? } $body);
        }
        #[allow(unreachable_code)]
        $vis fn $f ( $( $arg )* ) $( -> $retty )? {
            $crate::simd_dispatch!(@dispatch_cases ( $( $level )+ ) $f ( $( $arg )* ) );
            panic!("no suitable SIMD level");
        }
    };
    ( @cases [ $( $level:ident )* ] $vis:vis $f:ident $sig:tt $body:tt) => {
        $(
            $crate::simd_dispatch!(@level $level : $f $sig $body );
        )+
    };
    ( @level fallback : $f:ident { ( $( $args:tt ),* ) $( -> $retty:ty )? } $body:block ) => {
        #[inline]
        pub fn fallback ( $( $args ),* ) $( -> $retty )? {
            const HAS_NEON: bool = false;
            const HAS_NEON_FP16: bool = false;
            $body
        }
    };
    ( @level neon : $f:ident { ( $( $args:tt )* ) $( -> $retty:ty )? } $body:block ) => {
        #[target_feature(enable = "neon")]
        #[inline]
        pub fn neon ( $( $args )* ) $( -> $retty )? {
            use $crate::neon as simd;
            const HAS_NEON: bool = true;
            const HAS_NEON_FP16: bool = false;
            $body
        }
    };
    ( @level neon_fp16 : $f:ident { ( $( $args:tt )* ) $( -> $retty:ty )? } $body:block ) => {
        #[target_feature(enable = "neon", enable = "fp16")]
        #[inline]
        pub fn neon_fp16 ( $( $args )* ) $( -> $retty )? {
            use $crate::neon as simd;
            const HAS_NEON: bool = true;
            const HAS_NEON_FP16: bool = true;
            $body
        }
    };
    ( @dispatch_cases ( $( $level:ident )+ ) $f:ident $args:tt ) => {
        $(
            simd_dispatch!(@dispatch $level $f $args );
        )+
    };
    ( @dispatch neon $f:ident ( $( $arg:ident : $argty:ty ),* ) ) => {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Consider the possibilty of somebody abusing the private
            // macro syntax. I'm not sure if there's a good way to
            // avoid that (other than moving to proc macros).
            unsafe {
                return $f::neon( $( $arg ),* );
            }
        }
    };
    ( @dispatch neon_fp16 $f:ident ( $( $arg:ident : $argty:ty ),* ) ) => {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("fp16") {
            // Consider the possibility of somebody abusing the private
            // macro syntax. I'm not sure if there's a good way to
            // avoid that (other than moving to proc macros).
            unsafe {
                return $f::neon_fp16( $( $arg ),* );
            }
        }
    };
    ( @dispatch fallback $f:ident ( $( $arg:ident : $argty:ty ),* ) ) => {
        return $f::fallback( $( $arg ),* );
    };
}
