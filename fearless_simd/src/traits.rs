// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{Level, Simd};

pub trait Select<T> {
    fn select(self, if_true: T, if_false: T) -> T;
}

// Same as pulp
pub trait WithSimd {
    type Output;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output;
}

impl<R, F: FnOnce(Level) -> R> WithSimd for F {
    type Output = R;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        self(simd.level())
    }
}

pub trait Bytes: Sized {
    type Bytes;

    fn to_bytes(self) -> Self::Bytes;

    fn from_bytes(value: Self::Bytes) -> Self;

    fn bitcast<U: Bytes<Bytes = Self::Bytes>>(self) -> U {
        U::from_bytes(self.to_bytes())
    }
}

pub(crate) mod seal {
    pub trait Seal {}
}

/// Value conversion, adding a SIMD blessing.
///
/// Analogous to [`From`], but takes a SIMD token, which is used to bless
/// the new value. Most such conversions are safe transmutes, but this
/// trait also supports splats, and implementations can use the SIMD token
/// to use an efficient splat intrinsic.
///
/// The [`SimdInto`] trait is also provided for convenience.
pub trait SimdFrom<T, S: Simd> {
    fn simd_from(value: T, simd: S) -> Self;
}

/// Value conversion, adding a SIMD blessing.
///
/// This trait is syntactic sugar for [`SimdFrom`] and exists only to allow
/// `impl SimdInto` syntax in signatures, which would otherwise require
/// cumbersome `where` clauses in terms of `SimdFrom`.
///
/// Avoid implementing this trait directly, prefer implementing [`SimdFrom`].
pub trait SimdInto<T, S> {
    fn simd_into(self, simd: S) -> T;
}

impl<F, T: SimdFrom<F, S>, S: Simd> SimdInto<T, S> for F {
    fn simd_into(self, simd: S) -> T {
        SimdFrom::simd_from(self, simd)
    }
}

impl<T, S: Simd> SimdFrom<T, S> for T {
    fn simd_from(value: T, _simd: S) -> Self {
        value
    }
}

pub trait SimdElement {
    type Mask: SimdElement;
}

impl SimdElement for f32 {
    type Mask = i32;
}

impl SimdElement for f64 {
    type Mask = i64;
}

impl SimdElement for u8 {
    type Mask = i8;
}

impl SimdElement for i8 {
    type Mask = i8;
}

impl SimdElement for u16 {
    type Mask = i16;
}

impl SimdElement for i16 {
    type Mask = i16;
}

impl SimdElement for u32 {
    type Mask = i32;
}

impl SimdElement for i32 {
    type Mask = i32;
}

impl SimdElement for i64 {
    type Mask = i64;
}
