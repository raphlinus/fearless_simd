// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::fallback::Fallback;
use crate::arch::{Arch, fallback};
use crate::generic::{generic_combine, generic_op, generic_split};
use crate::ops::{OpSig, TyFlavor, ops_for_type, reinterpret_ty, valid_reinterpret};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

#[derive(Clone, Copy)]
pub struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Fallback"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub fn mk_fallback_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();

    quote! {
        use core::ops::*;
        use crate::{seal::Seal, Level, Simd, SimdInto};

        #imports

        #[cfg(all(feature = "libm", not(feature = "std")))]
        trait FloatExt {
            fn floor(self) -> f32;
            fn sqrt(self) -> f32;
        }

        #[cfg(all(feature = "libm", not(feature = "std")))]
        impl FloatExt for f32 {
            #[inline(always)]
            fn floor(self) -> f32 {
                libm::floorf(self)
            }
            #[inline(always)]
            fn sqrt(self) -> f32 {
                libm::sqrtf(self)
            }
        }

        /// The SIMD token for the "fallback" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Fallback {
            pub fallback: crate::core_arch::fallback::Fallback,
        }

        impl Fallback {
            #[inline]
            pub fn new() -> Self {
                Fallback {
                    fallback: crate::core_arch::fallback::Fallback::new(),
                }
            }
        }

        impl Seal for Fallback {}

        #simd_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        let scalar_bits = vec_ty.scalar_bits;
        let ty_name = vec_ty.rust_name();
        let ty = vec_ty.rust();
        for (method, sig) in ops_for_type(vec_ty, true) {
            if (vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow"))
                || vec_ty.n_bits() > 256
            {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }
            let method_name = format!("{method}_{ty_name}");
            let method_ident = Ident::new(&method_name, Span::call_site());
            let ret_ty = sig.ret_ty(vec_ty, TyFlavor::SimdTrait);
            let method = match sig {
                OpSig::Splat => {
                    let scalar = vec_ty.scalar.rust(scalar_bits);
                    let num_elements = vec_ty.len;
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, val: #scalar) -> #ret_ty {
                            [val; #num_elements].simd_into(self)
                        }
                    }
                }
                OpSig::Unary => {
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let args = [quote! { a[#idx] }];
                                let expr = Fallback.expr(method, vec_ty, &args);
                                quote! { #expr }
                            })
                            .collect::<Vec<_>>(),
                    );

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::WidenNarrow(t) => {
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let scalar_ty = t.scalar.rust(t.scalar_bits);
                                quote! { a[#idx] as #scalar_ty }
                            })
                            .collect::<Vec<_>>(),
                    );

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::Binary => {
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let b = if fallback::translate_op(method)
                                    .map(rhs_reference)
                                    .unwrap_or(true)
                                {
                                    quote! { &b[#idx] }
                                } else {
                                    quote! { b[#idx] }
                                };

                                let args = [quote! { a[#idx] }, quote! { #b }];
                                let expr = Fallback.expr(method, vec_ty, &args);
                                quote! { #expr }
                            })
                            .collect::<Vec<_>>(),
                    );

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::Shift => {
                    let arch_ty = Fallback.arch_ty(vec_ty);
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let args = [quote! { a[#idx] }, quote! { b as #arch_ty }];
                                let expr = Fallback.expr(method, vec_ty, &args);
                                quote! { #expr }
                            })
                            .collect::<Vec<_>>(),
                    );

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: u32) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::Ternary => {
                    if method == "madd" {
                        // TODO: This is has slightly different semantics than a fused multiply-add,
                        // since we are not actually fusing it, should this be documented?
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                               a.add(b.mul(c))
                            }
                        }
                    } else {
                        let args = [
                            quote! { a.into() },
                            quote! { b.into() },
                            quote! { c.into() },
                        ];

                        let expr = Fallback.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                               #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Compare => {
                    let mask_type = VecType::new(ScalarType::Mask, vec_ty.scalar_bits, vec_ty.len);
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let args = [quote! { &a[#idx] }, quote! { &b[#idx] }];
                                let expr = Fallback.expr(method, vec_ty, &args);
                                let mask_ty = mask_type.scalar.rust(scalar_bits);
                                quote! { #expr as #mask_ty }
                            })
                            .collect::<Vec<_>>(),
                    );

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::Select => {
                    let mask_ty = vec_ty.mask_ty().rust();
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                quote! { if a[#idx] != 0 { b[#idx] } else { c[#idx] } }
                            })
                            .collect::<Vec<_>>(),
                    );

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip => {
                    let (zip1, zip2) = match method {
                        "zip" => {
                            let zip1 = make_list(
                                (0..vec_ty.len / 2)
                                    .map(|idx| {
                                        quote! {a[#idx], b[#idx] }
                                    })
                                    .collect::<Vec<_>>(),
                            );

                            let zip2 = make_list(
                                (vec_ty.len / 2..vec_ty.len)
                                    .map(|idx| {
                                        quote! {a[#idx], b[#idx] }
                                    })
                                    .collect::<Vec<_>>(),
                            );

                            (zip1, zip2)
                        }
                        "unzip" => {
                            let len = vec_ty.len;

                            let unzip_a = {
                                let mut low = (0..len / 2)
                                    .map(|i| {
                                        quote! { a[#i * 2] }
                                    })
                                    .collect::<Vec<_>>();
                                let high = (0..len / 2)
                                    .map(|i| {
                                        quote! { b[#i * 2] }
                                    })
                                    .collect::<Vec<_>>();
                                low.extend(high);

                                make_list(low)
                            };

                            let unzip_b = {
                                let mut low = (0..len / 2)
                                    .map(|i| {
                                        quote! { a[#i * 2 + 1] }
                                    })
                                    .collect::<Vec<_>>();
                                let high = (0..len / 2)
                                    .map(|i| {
                                        quote! { b[#i * 2 + 1] }
                                    })
                                    .collect::<Vec<_>>();
                                low.extend(high);

                                make_list(low)
                            };

                            (unzip_a, unzip_b)
                        }
                        _ => todo!(),
                    };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            (
                                #zip1.simd_into(self),
                                #zip2.simd_into(self),
                            )
                        }
                    }
                }
                OpSig::Cvt(scalar, scalar_bits) => {
                    let to_ty = &VecType::new(scalar, scalar_bits, vec_ty.len);
                    let scalar = to_ty.scalar.rust(scalar_bits);
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                quote! { a[#idx] as #scalar }
                            })
                            .collect::<Vec<_>>(),
                    );
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            #items.simd_into(self)
                        }
                    }
                }
                OpSig::Reinterpret(scalar, scalar_bits) => {
                    if valid_reinterpret(vec_ty, scalar, scalar_bits) {
                        let to_ty = reinterpret_ty(vec_ty, scalar, scalar_bits).rust();

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                #to_ty {
                                    val: bytemuck::cast(a.val),
                                    simd: a.simd,
                                }
                            }
                        }
                    } else {
                        quote! {}
                    }
                }
            };
            methods.push(method);
        }
    }
    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                Level::#level_tok(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                f()
            }

            #( #methods )*
        }
    }
}

/// Whether the second argument of the function needs to be passed by reference.
fn rhs_reference(method: &str) -> bool {
    !matches!(method, "copysign" | "min" | "max")
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}
