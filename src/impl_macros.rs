// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros used by implementations

// Not all macros will be used by all implementations.
#![allow(unused_macros)]
#![allow(unused_imports)]

// Adapted from similar macro in pulp
macro_rules! delegate {
    ( $prefix:path : $(
        $(#[$attr: meta])*
        $(unsafe $($placeholder: lifetime)?)?
        fn $func: ident $(<$(const $generic: ident: $generic_ty: ty),* $(,)?>)?(
            $($arg: ident: $ty: ty),* $(,)?
        ) $(-> $ret: ty)?;
    )*) => {
        $(
            #[doc=concat!("See [`", stringify!($prefix), "::", stringify!($func), "`].")]
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
    ( $opfn:ident ( $( $arg:ident : $argty:ident ),* ) -> $ret:ident = $ns:ident . $intrinsic:ident ) => {
        #[inline(always)]
        fn $opfn( self, $( $arg: $argty<Self> ),* ) -> $ret<Self> {
            self.$ns.$intrinsic( $($arg.into() ),* ).simd_into(self)
        }
    };

    // Pattern used for SIMD comparisons
    ( $opfn:ident ( $( $arg:ident : $argty:ident ),* ) -> $ret:ident
        = $nsc:ident . $cast:ident ( $ns:ident . $intrinsic:ident )
    ) => {
        #[inline(always)]
        fn $opfn( self, $( $arg: $argty<Self> ),* ) -> $ret<Self> {
            self.$nsc.$cast(self.$ns.$intrinsic( $($arg.into() ),* )).simd_into(self)
        }
    };

    // Pattern used for select on Intel
    ( $opfn:ident ( $a:ident : $aty:ident, $b:ident : $bty:ident, $c:ident : $cty:ident ) -> $ret:ident
        = $intrinsic:ident ( c, b, $cast:ident(a) )
    ) => {
        #[inline(always)]
        fn $opfn( self, $a:$aty<Self>, $b:$bty<Self>, $c:$cty<Self> ) -> $ret<Self> {
            self.$intrinsic($c.into(), $b.into(), self.$cast($a.into())).simd_into(self)
        }
    };

    // Pattern used for select on Neon
    ( $opfn:ident ( $a:ident : $aty:ident, $b:ident : $bty:ident, $c:ident : $cty:ident ) -> $ret:ident
        = $ns:ident . $intrinsic:ident ( $nsc:ident . $cast:ident(a), b, c )
    ) => {
        #[inline(always)]
        fn $opfn( self, $a:$aty<Self>, $b:$bty<Self>, $c:$cty<Self> ) -> $ret<Self> {
            self.$ns.$intrinsic(self.$nsc.$cast($a.into()), $b.into(), $c.into()).simd_into(self)
        }
    };

    // Pattern used by mul_add on Neon
    ( $opfn:ident ( $a:ident : $aty:ident, $b:ident : $bty:ident, $c:ident : $cty:ident ) -> $ret:ident
        = $ns:ident . $intrinsic:ident ( c, a, b )
    ) => {
        #[inline(always)]
        fn $opfn( self, $a:$aty<Self>, $b:$bty<Self>, $c:$cty<Self> ) -> $ret<Self> {
            self.$ns.$intrinsic($c.into(), $a.into(), $b.into()).simd_into(self)
        }
    };
}
pub(crate) use impl_op;
