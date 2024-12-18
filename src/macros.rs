// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros used by implementations

macro_rules! impl_simd_from_into {
    ( $simd:ty, $arch:ty ) => {
        impl From<$arch> for $simd {
            #[inline]
            fn from(arch: $arch) -> Self {
                unsafe { core::mem::transmute(arch) }
            }
        }

        impl From<$simd> for $arch {
            #[inline]
            fn from(simd: $simd) -> Self {
                unsafe { core::mem::transmute(simd) }
            }
        }
    };
}
pub(crate) use impl_simd_from_into;

macro_rules! impl_unaryop {
    ($( $tf:literal ),* : $opfn:ident ( $ty:ty ) = $intrinsic:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn $opfn(a: $ty) -> $ty {
            unsafe {
                $intrinsic(a.into()).into()
            }
        }
    };
}
pub(crate) use impl_unaryop;

macro_rules! impl_binop {
    ($( $tf:literal ),* : $opfn:ident ( $ty:ty ) = $intrinsic:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> $ty {
            unsafe {
                $intrinsic(a.into(), b.into()).into()
            }
        }
    };
}
pub(crate) use impl_binop;

macro_rules! impl_ternary {
    ($( $tf:literal ),* : $opfn:ident ( $ty:ty ) = $intrinsic:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty, c: $ty) -> $ty {
            unsafe {
                $intrinsic(a.into(), b.into(), c.into()).into()
            }
        }
    };
    ($( $tf:literal ),* : $opfn:ident ( $ty:ty ) = $intrinsic:ident cab) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty, c: $ty) -> $ty {
            unsafe {
                $intrinsic(c.into(), a.into(), b.into()).into()
            }
        }
    };
}
pub(crate) use impl_ternary;

macro_rules! impl_cmp {
    ($( $tf:literal ),* : $opfn:ident ( $ty:ty ) = $intrinsic:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> <$ty as $crate::Simd>::Mask {
            unsafe {
                core::mem::transmute($intrinsic(a.into(), b.into()))
            }
        }
    };
}
pub(crate) use impl_cmp;

// This is the same as impl_binop but with a wilder transmute
macro_rules! impl_cmp_mask {
    ($( $tf:literal ),* : $opfn:ident ( $ty:ty ) = $intrinsic:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn $opfn(a: $ty, b: $ty) -> $ty {
            unsafe {
                core::mem::transmute($intrinsic(a.into(), b.into()))
            }
        }
    };
}
pub(crate) use impl_cmp_mask;

macro_rules! impl_cast {
    ($( $tf:literal ),* : $opfn:ident ( $from:ty) -> $to:ident = $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(a: $from) -> $to {
            unsafe {
                $intrinsic(a.into()).into()
            }
        }
    };
}
pub(crate) use impl_cast;

macro_rules! impl_select {
    ($( $tf:literal ),* : ( $ty:ty ) = $intrinsic:ident, $cast:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn select(a: <$ty as $crate::Simd>::Mask, b: $ty, c: $ty) -> $ty {
            unsafe {
                $intrinsic($cast(a.into()), b.into(), c.into()).into()
            }
        }
    };
    ($( mask $tf:literal ),* : ( $ty:ty ) = $intrinsic:ident, $cast:ident ) => {
        #[target_feature( $( enable = $tf ),* )]
        #[inline]
        pub fn select(a: $ty, b: $ty, c: $ty) -> $ty {
            unsafe {
                $intrinsic($cast(a.into()), b.into(), c.into()).into()
            }
        }
    };
}
pub(crate) use impl_select;
