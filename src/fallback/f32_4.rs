// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f32x4`] SIMD values.

use crate::f32x4;

pub fn splat(x: f32) -> f32x4 {
    [x, x, x, x].into()
}

macro_rules! impl_unaryop {
    ( $opfn:ident ) => {
        #[inline]
        pub fn $opfn(a: f32x4) -> f32x4 {
            [
                a.0[0].$opfn(),
                a.0[1].$opfn(),
                a.0[2].$opfn(),
                a.0[3].$opfn(),
            ]
            .into()
        }
    };
}

macro_rules! impl_binop {
    ( $opfn:ident ) => {
        #[inline]
        pub fn $opfn(a: f32x4, b: f32x4) -> f32x4 {
            [
                a.0[0].$opfn(b.0[0]),
                a.0[1].$opfn(b.0[1]),
                a.0[2].$opfn(b.0[2]),
                a.0[3].$opfn(b.0[3]),
            ]
            .into()
        }
    };
}

use std::ops::{Add, Div, Mul, Neg, Sub};

impl_unaryop!(sqrt);
impl_unaryop!(neg);
impl_unaryop!(abs);
impl_unaryop!(floor);
impl_unaryop!(ceil);
impl_unaryop!(round);
impl_binop!(add);
impl_binop!(sub);
impl_binop!(mul);
impl_binop!(div);
