// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros used by implementations

macro_rules! impl_simd_from_into {
    ( $simd:ty, $arch:ty ) => {
        impl From<$arch> for $simd {
            #[inline]
            fn from(arch: $arch) -> Self {
                unsafe {
                    core::mem::transmute(arch)
                }
            }
        }

        impl From<$simd> for $arch {
            #[inline]
            fn from(simd: $simd) -> Self {
                unsafe {
                    core::mem::transmute(simd)
                }
            }
        }
    };
}
pub(crate) use impl_simd_from_into;

macro_rules! impl_unaryop {
    ($( $tf:literal ),* : $opfn:ident, $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(self) -> Self {
            unsafe {
                // TODO: many of these transmutes can be replaced with .into()
                core::mem::transmute($intrinsic(core::mem::transmute(self)))
            }
        }
    };
}
pub(crate) use impl_unaryop;

macro_rules! impl_binop {
    ($( $tf:literal ),* : $opfn:ident, $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(self, rhs: Self) -> Self {
            unsafe {
                core::mem::transmute($intrinsic(
                    core::mem::transmute(self),
                    core::mem::transmute(rhs),
                ))
            }
        }
    };
}
pub(crate) use impl_binop;

macro_rules! impl_ternary {
    ($( $tf:literal ),* : $opfn:ident, $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(self, a: Self, b: Self) -> Self {
            unsafe {
                core::mem::transmute($intrinsic(
                    core::mem::transmute(self),
                    core::mem::transmute(a),
                    core::mem::transmute(b),
                ))
            }
        }
    };
}
pub(crate) use impl_ternary;

macro_rules! impl_cmp {
    ($( $tf:literal ),* : $opfn:ident, $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(self, rhs: Self) -> <Self as crate::Simd>::Mask {
            unsafe {
                core::mem::transmute($intrinsic(
                    core::mem::transmute(self),
                    core::mem::transmute(rhs),
                ))
            }
        }
    };
}
pub(crate) use impl_cmp;

macro_rules! impl_cmp_mask {
    ($( $tf:literal ),* : $opfn:ident, $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(self, rhs: Self) -> Self {
            unsafe {
                core::mem::transmute($intrinsic(
                    core::mem::transmute(self),
                    core::mem::transmute(rhs),
                ))
            }
        }
    };
}
pub(crate) use impl_cmp_mask;

macro_rules! impl_select {
    ($( $tf:literal ),* : $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn select<T: crate::Simd<Mask = Self>>(self, a: T, b: T) -> T {
            unsafe {
                T::from_mask(
                    core::mem::transmute($intrinsic(
                        core::mem::transmute(self),
                        core::mem::transmute(T::to_mask(a)),
                        core::mem::transmute(T::to_mask(b)),
                    ))
                )
            }
        }
    };
}
pub(crate) use impl_select;

macro_rules! impl_cast {
    ($( $tf:literal ),* : $opfn:ident -> $to:ident, $intrinsic:ident ) => {
        #[inline]
        #[target_feature( $( enable = $tf ),* )]
        pub fn $opfn(self) -> $to {
            unsafe {
                core::mem::transmute($intrinsic(core::mem::transmute(self)))
            }
        }
    };
}
pub(crate) use impl_cast;
