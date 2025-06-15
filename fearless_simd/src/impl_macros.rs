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
