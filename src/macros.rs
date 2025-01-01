// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta:meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[target_feature(enable = "neon")]
            #[inline]
            unsafe fn inner_neon(neon: $crate::aarch64::Neon $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( neon $( , $arg )* )
            }
            #[target_feature(enable = "neon,fp16")]
            #[inline]
            unsafe fn inner_fp16(fp16: $crate::aarch64::Fp16 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( fp16 $( , $arg )* )
            }
            match level {
                Level::Fallback(fb) => $inner(fb $( , $arg )* ),
                Level::Neon(neon) => unsafe { inner_neon (neon $( , $arg )* ) }
                Level::Fp16(fp16) => unsafe { inner_fp16 (fp16 $( , $arg )* ) }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta:meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[target_feature(enable = "avx2,bmi2,f16c,fma,lzcnt")]
            #[inline]
            unsafe fn inner_avx2(avx2: $crate::x86_64::Avx2 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( avx2 $( , $arg )* )
            }
            match level {
                Level::Fallback(fb) => $inner(fb $( , $arg )* ),
                Level::Avx2(avx2) => unsafe { inner_avx2 (avx2 $( , $arg )* ) }
            }
        }
    };
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta:meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            match level {
                Level::Fallback(fb) => $inner(fb $( , $arg )* ),
            }
        }
    };
}
