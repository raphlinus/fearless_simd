// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operations on [`f32x4`] SIMD values.

use crate::{f32x4, mask32x4};

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
impl_unaryop!(trunc);
impl_unaryop!(round);
impl_unaryop!(round_ties_even);
impl_binop!(add);
impl_binop!(sub);
impl_binop!(mul);
impl_binop!(div);
impl_binop!(copysign);
impl_binop!(powf);

pub fn mul_add(a: f32x4, b: f32x4, c: f32x4) -> f32x4 {
    [
        a.0[0].mul_add(b.0[0], c.0[0]),
        a.0[1].mul_add(b.0[1], c.0[1]),
        a.0[2].mul_add(b.0[2], c.0[2]),
        a.0[3].mul_add(b.0[3], c.0[3]),
    ]
    .into()
}

fn mask(b: bool) -> i32 {
    -(b as i32)
}

pub fn simd_gt(a: f32x4, b: f32x4) -> mask32x4 {
    [
        mask(a.0[0] > b.0[0]),
        mask(a.0[1] > b.0[1]),
        mask(a.0[2] > b.0[2]),
        mask(a.0[3] > b.0[3]),
    ]
    .into()
}

pub fn simd_ge(a: f32x4, b: f32x4) -> mask32x4 {
    [
        mask(a.0[0] >= b.0[0]),
        mask(a.0[1] >= b.0[1]),
        mask(a.0[2] >= b.0[2]),
        mask(a.0[3] >= b.0[3]),
    ]
    .into()
}

fn sel1(a: i32, b: f32, c: f32) -> f32 {
    if a < 0 { b } else { c }
}

pub fn select(a: mask32x4, b: f32x4, c: f32x4) -> f32x4 {
    [
        sel1(a.0[0], b.0[0], c.0[0]),
        sel1(a.0[1], b.0[1], c.0[1]),
        sel1(a.0[2], b.0[2], c.0[2]),
        sel1(a.0[3], b.0[3], c.0[3]),
    ]
    .into()
}

pub fn copy_lane<const LANE1: i32, const LANE2: i32>(a: f32x4, b: f32x4) -> f32x4 {
    let mut result = a.0;
    result[LANE1 as usize] = b.0[LANE2 as usize];
    result.into()
}
