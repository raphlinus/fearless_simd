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
        $( #[$meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[target_feature(enable = "neon")]
            #[inline]
            unsafe fn inner_neon(neon: $crate::aarch64::Neon $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( neon $( , $arg )* )
            }
            //#[target_feature(enable = "neon,fp16")]
            //#[inline]
            //unsafe fn inner_fp16(fp16: $crate::aarch64::Fp16 $( , $arg: $ty )* ) $( -> $ret )? {
            //    $inner( fp16 $( , $arg )* )
            //}
            match level {
                //Level::Fallback(fb) => $inner(fb $( , $arg )* ),
                Level::Neon(neon) => unsafe { inner_neon (neon $( , $arg )* ) }
                //Level::Fp16(fp16) => unsafe { inner_fp16 (fp16 $( , $arg )* ) }
            }
        }
    };
}

#[macro_export]
macro_rules! simd_dispatch_explicit {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident $args:tt -> $ret:ty
        = match level {
            $(
                $level:ident => $inner:ident,
            )*
        }
    ) => {
        $( #[$meta] )* $vis
        simd_dispatch_explicit!(@funny $func lv $args -> $ret {
            $(
                simd_dispatch_explicit!(@case lv $level => $inner $args -> $ret);
            )*
            unreachable!()
        });
    };
    (
        @funny $func:ident $lv: ident ( $( $arg:ident : $ty:ty ),* ) $( -> $ret:ty )? $body:block
    ) => {
        fn $func( $lv: $crate::Level $(, $arg: $ty )* ) $( -> $ret )? $body
    };
    (
        @case $lv:ident neon => $inner:ident ( $( $arg:ident : $ty:ty $(,)? ),* ) $( -> $ret:ty )?
    ) => {
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn inner_neon(neon: $crate::aarch64::Neon $( , $arg: $ty )* ) $( -> $ret )? {
            $inner( neon $( , $arg )* )
        }
        if let Some(neon) = $lv.as_neon() {
            return unsafe { inner_neon( neon $(, $arg )* ) };
        }
    }
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
