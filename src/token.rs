// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An experimental safe token

use core::arch::aarch64;
use std::arch::is_aarch64_feature_detected;

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub enum Aarch64 {
    Fallback,
    Neon(Neon),
}

impl Aarch64 {
    pub fn new() -> Self {
        if is_aarch64_feature_detected!("neon") {
            Aarch64::Neon(Neon { _private: () })
        } else {
            Aarch64::Fallback
        }
    }

    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_neon<W: WithSimd>(f: W, n: Neon) -> W::Output {
            f.with_simd(n)
        }
        match self {
            Aarch64::Fallback => f.with_simd(()),
            Aarch64::Neon(n) => unsafe { dispatch_neon(f, n) },
        }
    }

    // This kind of function is stable even when more levels
    // are added, which matching on the enum would not be.
    pub fn as_neon(self) -> Option<Neon> {
        if let Aarch64::Neon(n) = self {
            Some(n)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
pub struct Neon {
    _private: (),
}

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

use core::arch::aarch64::*;

impl Neon {
    delegate! {
        fn vaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
        fn vmulq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
        fn vfmaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
        fn vdupq_n_f32(a: f32) -> float32x4_t;
        fn vabsq_f32(a: float32x4_t) -> float32x4_t;
        fn vcgtq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t;
        fn vreinterpretq_s32_u32(a: uint32x4_t) -> int32x4_t;
        fn vreinterpretq_u32_s32(a: int32x4_t) -> uint32x4_t;
        fn vbslq_f32(a: uint32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
		fn vsqrtq_f32(a: float32x4_t) -> float32x4_t;
    }

    #[inline(always)]
    pub fn call<T, R>(self, f: impl FnOnce(T) -> R, arg: T) -> R {
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn inner<T, R>(f: impl FnOnce(T) -> R, arg: T) -> R {
            f(arg)
        }
        unsafe {
            inner(f, arg)
        }
    }
}

// Probably Send and 'static also, maybe Debug
pub trait Simd: Sized + Clone + Copy {
    fn arch(self) -> Aarch64;
    fn splat_f32x4(self, val: f32) -> f32x4<Self>;
    fn add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn mul_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn mul_add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self>;
    fn abs_f32x4(self, a: f32x4<Self>) -> f32x4<Self>;
    fn simd_gt_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> mask32x4<Self>;
    fn select_f32x4(self, a: mask32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self>;
}

// Same as pulp
pub trait WithSimd {
    type Output;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output;
}

impl<R, F: FnOnce(Aarch64) -> R> WithSimd for F {
    type Output = R;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        self(simd.arch())
    }
}

pub trait IntoSimdVec<T, S> {
    fn into_simd(self, simd: S) -> T;
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct f32x4<S: Simd> {
    pub val: [f32; 4],
    pub simd: S,
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct mask32x4<S: Simd> {
    pub val: [i32; 4],
    pub simd: S,
}

impl<S: Simd, T> IntoSimdVec<T, S> for T {
    fn into_simd(self, _simd: S) -> T {
        self
    }
}

impl<S: Simd> IntoSimdVec<f32x4<S>, S> for [f32; 4] {
    fn into_simd(self, simd: S) -> f32x4<S> {
        f32x4 { val: self, simd }
    }
}

impl<S: Simd> IntoSimdVec<f32x4<S>, S> for core::arch::aarch64::float32x4_t {
    fn into_simd(self, simd: S) -> f32x4<S> {
        unsafe {
            f32x4 {
                val: core::mem::transmute(self),
                simd,
            }
        }
    }
}

impl<S: Simd> IntoSimdVec<f32x4<S>, S> for f32 {
    fn into_simd(self, simd: S) -> f32x4<S> {
        simd.splat_f32x4(self)
    }
}

impl<S: Simd> IntoSimdVec<mask32x4<S>, S> for [i32; 4] {
    fn into_simd(self, simd: S) -> mask32x4<S> {
        mask32x4 { val: self, simd }
    }
}

impl<S: Simd> IntoSimdVec<mask32x4<S>, S> for core::arch::aarch64::int32x4_t {
    fn into_simd(self, simd: S) -> mask32x4<S> {
        unsafe {
            mask32x4 {
                val: core::mem::transmute(self),
                simd,
            }
        }
    }
}

fn mask(b: bool) -> i32 {
    -(b as i32)
}

fn sel1(a: i32, b: f32, c: f32) -> f32 {
    if a < 0 { b } else { c }
}

impl Simd for () {
    #[inline]
    fn arch(self) -> Aarch64 {
        Aarch64::Fallback
    }

    #[inline]
    fn splat_f32x4(self, val: f32) -> f32x4<Self> {
        [val; 4].into_simd(self)
    }

    #[inline]
    fn add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0] + b.val[0],
            a.val[1] + b.val[1],
            a.val[2] + b.val[2],
            a.val[3] + b.val[3],
        ];
        val.into_simd(self)
    }

    #[inline]
    fn mul_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0] * b.val[0],
            a.val[1] * b.val[1],
            a.val[2] * b.val[2],
            a.val[3] * b.val[3],
        ];
        val.into_simd(self)
    }

    #[inline]
    fn mul_add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        let val = [
            a.val[0].mul_add(b.val[0], c.val[0]),
            a.val[1].mul_add(b.val[1], c.val[1]),
            a.val[2].mul_add(b.val[2], c.val[2]),
            a.val[3].mul_add(b.val[3], c.val[3]),
        ];
        val.into_simd(self)
    }

    #[inline]
    fn abs_f32x4(self, a: f32x4<Self>) -> f32x4<Self> {
        a.val.map(f32::abs).into_simd(self)
    }

    #[inline]
    fn simd_gt_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> mask32x4<Self> {
        [
            mask(a.val[0] > b.val[0]),
            mask(a.val[1] > b.val[1]),
            mask(a.val[2] > b.val[2]),
            mask(a.val[3] > b.val[3]),
        ]
        .into_simd(self)
    }

    #[inline]
    fn select_f32x4(self, a: mask32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        [
            sel1(a.val[0], b.val[0], c.val[0]),
            sel1(a.val[1], b.val[1], c.val[1]),
            sel1(a.val[2], b.val[2], c.val[2]),
            sel1(a.val[3], b.val[3], c.val[3]),
        ]
        .into_simd(self)
    }
}

impl<S: Simd> From<f32x4<S>> for [f32; 4] {
    #[inline(always)]
    fn from(value: f32x4<S>) -> Self {
        value.val
    }
}

impl<S: Simd> From<f32x4<S>> for aarch64::float32x4_t {
    #[inline(always)]
    fn from(value: f32x4<S>) -> Self {
        unsafe { core::mem::transmute(value.val) }
    }
}

impl<S: Simd> From<mask32x4<S>> for aarch64::int32x4_t {
    #[inline(always)]
    fn from(value: mask32x4<S>) -> Self {
        unsafe { core::mem::transmute(value.val) }
    }
}

macro_rules! impl_op {
    ($opfn:ident ( $( $arg:ident : $argty:ident ),* ) -> $ret:ident = $intrinsic:ident ) => {
        #[inline(always)]
        fn $opfn( self, $( $arg: $argty<Self> ),* ) -> $ret<Self> {
            self.$intrinsic( $($arg.into() ),* ).into_simd(self)
        }
    };
}

impl Simd for Neon {
    #[inline(always)]
    fn arch(self) -> Aarch64 {
        Aarch64::Neon(self)
    }

    #[inline(always)]
    fn splat_f32x4(self, x: f32) -> f32x4<Self> {
        self.vdupq_n_f32(x).into_simd(self)
    }

    impl_op!(add_f32x4(a: f32x4, b: f32x4) -> f32x4 = vaddq_f32);
    impl_op!(mul_f32x4(a: f32x4, b: f32x4) -> f32x4 = vaddq_f32);
    impl_op!(mul_add_f32x4(a: f32x4, b: f32x4, c: f32x4) -> f32x4 = vfmaq_f32);
    impl_op!(abs_f32x4(a: f32x4) -> f32x4 = vabsq_f32);

    #[inline(always)]
    fn simd_gt_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> mask32x4<Self> {
        self.vreinterpretq_s32_u32(self.vcgtq_f32(a.into(), b.into()))
            .into_simd(self)
    }

    #[inline(always)]
    fn select_f32x4(self, a: mask32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self> {
        self.vbslq_f32(self.vreinterpretq_u32_s32(a.into()), b.into(), c.into())
            .into_simd(self)
    }
}

impl<S: Simd> core::ops::Add for f32x4<S> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        self.simd.add_f32x4(self, rhs)
    }
}

impl<S: Simd> core::ops::Add<f32> for f32x4<S> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: f32) -> Self::Output {
        self.simd.add_f32x4(self, rhs.into_simd(self.simd))
    }
}

impl<S: Simd> core::ops::Add<f32x4<S>> for f32 {
    type Output = f32x4<S>;

    #[inline(always)]
    fn add(self, rhs: f32x4<S>) -> Self::Output {
        rhs.simd.add_f32x4(self.into_simd(rhs.simd), rhs)
    }
}

impl<S: Simd> core::ops::Mul for f32x4<S> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.simd.mul_f32x4(self, rhs)
    }
}

impl<S: Simd> core::ops::Mul<f32> for f32x4<S> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        self.simd.mul_f32x4(self, rhs.into_simd(self.simd))
    }
}

impl<S: Simd> core::ops::Mul<f32x4<S>> for f32 {
    type Output = f32x4<S>;

    #[inline(always)]
    fn mul(self, rhs: f32x4<S>) -> Self::Output {
        rhs.simd.mul_f32x4(self.into_simd(rhs.simd), rhs)
    }
}

impl<S: Simd> f32x4<S> {
    #[inline(always)]
    pub fn mul_add(
        self,
        b: impl IntoSimdVec<f32x4<S>, S>,
        c: impl IntoSimdVec<f32x4<S>, S>,
    ) -> f32x4<S> {
        self.simd
            .mul_add_f32x4(self, b.into_simd(self.simd), c.into_simd(self.simd))
    }

    #[inline(always)]
    pub fn abs(self) -> f32x4<S> {
        self.simd.abs_f32x4(self)
    }

    #[inline(always)]
    pub fn simd_gt(self, b: impl IntoSimdVec<f32x4<S>, S>) -> mask32x4<S> {
        self.simd.simd_gt_f32x4(self, b.into_simd(self.simd))
    }
}

pub trait Select<T> {
    fn select(self, if_true: T, if_false: T) -> T;
}

impl<S: Simd> Select<f32x4<S>> for mask32x4<S> {
    fn select(self, if_true: f32x4<S>, if_false: f32x4<S>) -> f32x4<S> {
        self.simd.select_f32x4(self, if_true, if_false)
    }
}
