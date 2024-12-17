// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub trait Simd: Bytes {
    const LEN: usize;

    type Element;

    type Mask: Mask;

    fn to_mask(self) -> Self::Mask;

    fn from_mask(value: Self::Mask) -> Self;
}

pub trait Mask: Bytes {
    const LEN: usize;

    type Element;

    type Bytes: Simd;
}

pub trait Bytes {
    type Bytes: Simd;

    fn to_bytes(self) -> Self::Bytes;

    fn from_bytes(value: Self::Bytes) -> Self;
}

macro_rules! impl_basetype {
    ($simd:ident, $element:ty, $n:expr) => {
        // TODO: consider Debug, but our f16 doesn't
        #[derive(Clone, Copy, Default)]
        #[repr(transparent)]
        pub struct $simd(pub [$element; $n]);

        impl $simd {
            #[inline]
            pub const fn len(&self) -> usize {
                $n
            }

            #[inline]
            pub const fn as_array(&self) -> &[$element; $n] {
                &self.0
            }

            #[inline]
            pub fn as_mut_array(&mut self) -> &mut [$element; $n] {
                &mut self.0
            }

            #[inline]
            pub const unsafe fn load(ptr: *const [$element; $n]) -> Self {
                // This is adapted from std::simd but all this is almost certainly
                // not needed, because the alignment matches.
                unsafe {
                    let mut tmp = core::mem::MaybeUninit::<Self>::uninit();
                    core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr().cast(), 1);
                    tmp.assume_init()
                }
            }

            #[inline]
            pub unsafe fn store(self, ptr: *mut [$element; $n]) {
                // Same comment as for `load`
                unsafe {
                    let tmp = self;
                    core::ptr::copy_nonoverlapping(tmp.as_array(), ptr, 1);
                }
            }

            #[inline]
            pub const fn from_array(array: [$element; $n]) -> Self {
                Self(array)
            }

            #[inline]
            pub fn to_array(self) -> [$element; $n] {
                self.0
            }

            #[must_use]
            #[inline]
            pub fn from_slice(slice: &[$element]) -> Self {
                assert!(
                    slice.len() >= $n,
                    "slice length must be at least the number of elements"
                );
                unsafe { Self::load(slice.as_ptr().cast()) }
            }
        }

        impl From<[$element; $n]> for $simd {
            #[inline]
            fn from(array: [$element; $n]) -> Self {
                Self(array)
            }
        }

        impl From<$simd> for [$element; $n] {
            #[inline]
            fn from(vector: $simd) -> Self {
                vector.0
            }
        }

        // TODO: TryFrom impls, following std::simd
    };
}

macro_rules! impl_simd {
    ($simd:ident, $element:ty, $n:expr, $mask:ty, $bytes:ty) => {
        impl_basetype!($simd, $element, $n);
        impl $crate::Bytes for $simd {
            type Bytes = $bytes;

            /// A bitcast into the bytes type, which must be the same size.
            fn to_bytes(self) -> Self::Bytes {
                unsafe { core::mem::transmute(self) }
            }

            /// A bitcast from the bytes type, which must be the same size.
            fn from_bytes(bytes: Self::Bytes) -> Self {
                unsafe { core::mem::transmute(bytes) }
            }
        }

        impl $crate::Simd for $simd {
            const LEN: usize = $n;

            type Element = $element;

            type Mask = $mask;

            /// A bitcast into the mask type, which must be the same size.
            fn to_mask(self) -> <Self as $crate::Simd>::Mask {
                unsafe { core::mem::transmute(self) }
            }

            /// A bitcast from the mask type, which must be the same size.
            fn from_mask(mask: <Self as $crate::Simd>::Mask) -> Self {
                unsafe { core::mem::transmute(mask) }
            }
        }
    };
}

macro_rules! impl_mask {
    ($simd:ident, $element:ty, $n:expr, $bytes:ty) => {
        impl_basetype!($simd, $element, $n);
        impl $crate::Bytes for $simd {
            type Bytes = $bytes;

            /// A bitcast into the bytes type, which must be the same size.
            fn to_bytes(self) -> Self::Bytes {
                unsafe { core::mem::transmute(self) }
            }

            /// A bitcast from the bytes type, which must be the same size.
            fn from_bytes(bytes: Self::Bytes) -> Self {
                unsafe { core::mem::transmute(bytes) }
            }
        }

        impl $crate::Mask for $simd {
            const LEN: usize = $n;

            type Element = $element;

            type Bytes = $bytes;
        }
    };
}

// 64 bit types
impl_simd!(u16x4, u16, 4, mask16x4, u8x8);
impl_simd!(u8x8, u8, 8, mask8x8, u8x8);
impl_mask!(mask16x4, i16, 4, u8x8);
impl_mask!(mask8x8, i8, 8, u8x8);

// 128 bit types
impl_simd!(f32x4, f32, 4, mask32x4, u8x16);
impl_simd!(u32x4, u32, 4, mask32x4, u8x16);
impl_simd!(u16x8, u16, 8, mask16x8, u8x16);
impl_simd!(u8x16, u8, 16, mask8x16, u8x16);
impl_mask!(mask8x16, i8, 16, u8x16);
impl_mask!(mask16x8, i16, 8, u8x16);
impl_mask!(mask32x4, i32, 4, u8x16);

// 256 bit types
impl_simd!(f32x8, f32, 8, mask32x8, u8x32);
impl_simd!(u8x32, u8, 32, mask8x32, u8x32);
impl_mask!(mask8x32, i8, 32, u8x32);
impl_mask!(mask32x8, i32, 8, u8x32);

#[cfg(target_arch = "aarch64")]
mod f16;
pub use f16::*;

pub trait Bitcast<T>: Sized {
    fn bitcast(self) -> T;
}

impl<T: Bytes, U: Bytes<Bytes = T::Bytes>> Bitcast<U> for T {
    fn bitcast(self) -> U {
        U::from_bytes(self.to_bytes())
    }
}
