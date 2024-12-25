// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros used by implementations

// Not all macros will be used by all implementations.
#![allow(unused_macros)]
#![allow(unused_imports)]

// Adapted from similar macro in pulp
macro_rules! delegate {
    ( $(
        $(#[$attr: meta])*
        $(unsafe $($placeholder: lifetime)?)?
        fn $func: ident $(<$(const $generic: ident: $generic_ty: ty),* $(,)?>)?(
            $($arg: ident: $ty: ty),* $(,)?
        ) $(-> $ret: ty)?;
    )*) => {
        $(
            $(#[$attr])*
            #[inline(always)]
            pub $(unsafe $($placeholder)?)?
            fn $func $(<$(const $generic: $generic_ty),*>)?(self, $($arg: $ty),*) $(-> $ret)? {
                unsafe { $func $(::<$($generic,)*>)?($($arg,)*) }
            }
        )*
    };
}
pub(crate) use delegate;

macro_rules! impl_simd_from_into {
    ( $simd:ident, $arch:ty ) => {
        impl<S: Simd> SimdFrom<$arch, S> for $simd<S> {
            #[inline]
            fn simd_from(arch: $arch, simd: S) -> Self {
                $simd {
                    val: unsafe { core::mem::transmute(arch) },
                    simd,
                }
            }
        }

        impl<S: Simd> From<$simd<S>> for $arch {
            #[inline]
            fn from(value: $simd<S>) -> Self {
                unsafe { core::mem::transmute(value.val) }
            }
        }
    };
}
pub(crate) use impl_simd_from_into;

macro_rules! impl_op {
    ($opfn:ident ( $( $arg:ident : $argty:ident ),* ) -> $ret:ident = $intrinsic:ident ) => {
        #[inline(always)]
        fn $opfn( self, $( $arg: $argty<Self> ),* ) -> $ret<Self> {
            self.$intrinsic( $($arg.into() ),* ).simd_into(self)
        }
    };

    ($opfn:ident ( $( $arg:ident : $argty:ident ),* ) -> $ret:ident = $cast:ident ( $intrinsic:ident ) ) => {
        #[inline(always)]
        fn $opfn( self, $( $arg: $argty<Self> ),* ) -> $ret<Self> {
            self.$cast(self.$intrinsic( $($arg.into() ),* )).simd_into(self)
        }
    };
}
pub(crate) use impl_op;
