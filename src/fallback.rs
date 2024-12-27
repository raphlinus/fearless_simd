// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fallback implementations of SIMD methods.

// pub mod f32_4;

// Discussion question: what's the natural width? Most other implementations
// implementations default to 1, but perhaps we want to express the idea
// of operating on larger chunks of data, and encouraging autovectorization.
// pub use f32_4 as f32s;

use std::ops::{Add, Div, Mul, Sub};

use crate::{f32x4, mask32x4, seal::Seal, Level, Simd, SimdInto};

/// A SIMD token representing portable fallback.
///
/// Unlike other tokens, this one can always be constructed safely.
#[derive(Clone, Copy, Debug)]
pub struct Fallback {
    _private: (),
}

impl Fallback {
    pub fn new() -> Self {
        Fallback { _private: () }
    }
}

macro_rules! impl_opx4 {
    ( $opfn:ident ($a:ident : $aty:ident $(, $arg:ident: $ty:ident )* ) $( -> $rty:ident )?
        = $inner:ident ) => {
        #[inline]
        fn $opfn(self, $a: $aty<Self> $(, $arg: $ty<Self> )* ) $( -> $rty<Self> )? {
            [
                $a.val[0].$inner( $( $arg.val[0] ),* ),
                $a.val[1].$inner( $( $arg.val[1] ),* ),
                $a.val[2].$inner( $( $arg.val[2] ),* ),
                $a.val[3].$inner( $( $arg.val[3] ),* ),
            ]
            .simd_into(self)
        }
    };

}

macro_rules! impl_cmpx4 {
    ( $opfn:ident ($a:ident : $aty:ident $(, $arg:ident: $ty:ident )* ) $( -> $rty:ident )?
        = $inner:ident ) => {
        #[inline]
        fn $opfn(self, $a: $aty<Self> $(, $arg: $ty )* <Self>) $( -> $rty<Self> )? {
            [
                mask($a.val[0].$inner( $( & $arg.val[0] ),* )),
                mask($a.val[1].$inner( $( & $arg.val[1] ),* )),
                mask($a.val[2].$inner( $( & $arg.val[2] ),* )),
                mask($a.val[3].$inner( $( & $arg.val[3] ),* )),
            ]
            .simd_into(self)
        }
    };
}

fn mask(b: bool) -> i32 {
    -(b as i32)
}

fn sel1(a: i32, b: f32, c: f32) -> f32 {
    if a < 0 {
        b
    } else {
        c
    }
}

impl Seal for Fallback {}

impl Simd for Fallback {
    fn level(self) -> Level {
        Level::Fallback(self)
    }

    #[inline]
    fn splat_f32x4(self, val: f32) -> f32x4<Self> {
        [val; 4].simd_into(self)
    }

    #[inline]
    fn mul_add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0].mul_add(b.val[0], c.val[0]),
            a.val[1].mul_add(b.val[1], c.val[1]),
            a.val[2].mul_add(b.val[2], c.val[2]),
            a.val[3].mul_add(b.val[3], c.val[3]),
        ];
        val.simd_into(self)
    }

    impl_opx4!(add_f32x4(a: f32x4, b: f32x4) -> f32x4 = add);
    impl_opx4!(sub_f32x4(a: f32x4, b: f32x4) -> f32x4 = sub);
    impl_opx4!(mul_f32x4(a: f32x4, b: f32x4) -> f32x4 = mul);
    impl_opx4!(div_f32x4(a: f32x4, b: f32x4) -> f32x4 = div);
    impl_opx4!(copysign_f32x4(a: f32x4, b: f32x4) -> f32x4 = copysign);
    impl_opx4!(abs_f32x4(a: f32x4) -> f32x4 = abs);
    impl_opx4!(sqrt_f32x4(a: f32x4) -> f32x4 = sqrt);
    impl_cmpx4!(simd_gt_f32x4(a: f32x4, b: f32x4) -> mask32x4 = gt);

    #[inline]
    fn select_f32x4(self, a: mask32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        [
            sel1(a.val[0], b.val[0], c.val[0]),
            sel1(a.val[1], b.val[1], c.val[1]),
            sel1(a.val[2], b.val[2], c.val[2]),
            sel1(a.val[3], b.val[3], c.val[3]),
        ]
        .simd_into(self)
    }
}
