// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::types::{SIMD_TYPES, ScalarType, type_imports};

pub fn mk_ops() -> TokenStream {
    let imports = type_imports();

    let mut impls = vec![];

    for ty in SIMD_TYPES {
        let ops = match ty.scalar {
            ScalarType::Float => &["add", "sub", "mul", "div"][..],
            ScalarType::Int | ScalarType::Unsigned => &["add"],
            ScalarType::Mask => &["and", "or", "xor", "not"],
        };
        let simd = ty.rust();
        for op in ops {
            let is_bit = matches!(*op, "and" | "or" | "xor");
            let is_unary = matches!(*op, "neg" | "not");
            let mut trait_name = format!("{}{}", op[0..1].to_ascii_uppercase(), &op[1..]);
            let opfn = if is_bit {
                trait_name = format!("Bit{trait_name}");
                Ident::new(&format!("bit{op}"), Span::call_site())
            } else {
                Ident::new(op, Span::call_site())
            };
            let trait_id = Ident::new(&trait_name, Span::call_site());
            let simd_fn_name = format!("{op}_{}", ty.rust_name());
            let simd_fn = Ident::new(&simd_fn_name, Span::call_site());
            if is_unary {
                impls.push(quote! {
                    impl<S: Simd> core::ops::#trait_id for #simd<S> {
                        type Output = Self;
                        #[inline(always)]
                        fn #opfn(self) -> Self::Output {
                            self.simd.#simd_fn(self)
                        }
                    }
                });
            } else {
                impls.push(quote! {
                    impl<S: Simd> core::ops::#trait_id for #simd<S> {
                        type Output = Self;
                        #[inline(always)]
                        fn #opfn(self, rhs: Self) -> Self::Output {
                            self.simd.#simd_fn(self, rhs)
                        }
                    }
                });
                if ty.scalar != ScalarType::Mask {
                    let scalar = ty.scalar.rust(ty.scalar_bits);
                    impls.push(quote! {

                        impl<S: Simd> core::ops::#trait_id<#scalar> for #simd<S> {
                            type Output = Self;
                            #[inline(always)]
                            fn #opfn(self, rhs: #scalar) -> Self::Output {
                                self.simd.#simd_fn(self, rhs.simd_into(self.simd))
                            }
                        }

                        impl<S: Simd> core::ops::#trait_id<#simd<S>> for #scalar {
                            type Output = #simd<S>;
                            #[inline(always)]
                            fn #opfn(self, rhs: #simd<S>) -> Self::Output {
                                rhs.simd.#simd_fn(self.simd_into(rhs.simd), rhs)
                            }
                        }
                    });
                }
            }
        }
    }

    quote! {
        use crate::{Simd, SimdInto};
        #imports
        #( #impls )*
    }
}
