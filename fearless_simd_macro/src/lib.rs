// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::ToTokens;
use syn::{
    AttrStyle, Attribute, FnArg, GenericParam, Ident, LitBool, LitStr, Meta, Token, parenthesized,
    parse::{Parse, Parser},
    punctuated::Punctuated,
    spanned::Spanned,
    token::Paren,
    visit_mut::VisitMut,
};

type AttributeArgs = Punctuated<AttributeArg, Token![,]>;

struct AttributeArg {
    key: Ident,
    _equals: Token![=],
    value: AttributeValue,
}

#[derive(Debug)]
enum AttributeValue {
    String(LitStr),
    Bool(LitBool),
}

/// Return the architecture corresponding to the level.
///
/// String is empty if no arch support is needed. Intentionally a
/// curated set for now.
fn arch_for_level(level: &str) -> Option<&'static str> {
    Some(match level {
        "neon" => "aarch64",
        "fp16" => "aarch64",
        "avx2" => "x86_64",
        _ => return None,
    })
}

/// Enumerate the target_features required to support the given level.
///
/// Note that implied features need not be in this list; this list is
/// primarily used to generate a `#[target_feature]` attribute on
/// instances.
fn features_for_level(level: &str) -> &'static [&'static str] {
    match level {
        "neon" => &["neon"],
        "fp16" => &["fp16"],
        "avx2" => &["avx2"],
        _ => &[],
    }
}

/// Return the fearless_simd module for the given level.
fn mod_for_level(level: &str) -> &'static str {
    match level {
        "neon" | "fp16" => "neon",
        "avx2" => "avx2",
        "fallback" => "fallback",
        _ => panic!("no module found for level {level}"),
    }
}

fn detect_macro_for_arch(arch: &str) -> &'static str {
    match arch {
        "aarch64" => "is_aarch64_feature_detected",
        "x86_64" => "is_x86_feature_detected",
        // Note: wasm doesn't have a feature detection macro, that will
        // need to be handled elsewhere. See:
        // https://doc.rust-lang.org/core/arch/wasm32/index.html#simd
        _ => unreachable!("no feature macro known for {arch}"),
    }
}

#[proc_macro_attribute]
pub fn simd_dispatch(args: TokenStream, input: TokenStream) -> TokenStream {
    use quote::quote;
    use syn::{Item, Signature, parse_macro_input};

    let mut levels: Vec<String> = vec![];
    let mut opt_level_span = None;
    let mut do_module = false;
    for arg in AttributeArgs::parse_terminated.parse(args).unwrap() {
        let key = arg.key.to_string();
        match key.as_str() {
            "levels" => {
                if let AttributeValue::String(s) = &arg.value {
                    opt_level_span = Some(s.span());
                    levels = s
                        .value()
                        .split(',')
                        .map(|raw_level| raw_level.trim().to_owned())
                        .collect();
                } else {
                    panic!("levels must be comma-separated string");
                }
            }
            "module" => {
                if let AttributeValue::Bool(b) = &arg.value {
                    do_module = b.value;
                } else {
                    panic!("module takes a bool value");
                }
            }
            _ => panic!("unknown argument key {key}"),
        }
    }
    let level_span = opt_level_span.expect("must specify levels");

    let Item::Fn(func) = parse_macro_input!(input as Item) else {
        panic!("simd_dispatch not applied to a function");
    };
    let attrs = &func.attrs;
    let vis = &func.vis;
    let block = &func.block;
    let Signature {
        constness,
        asyncness,
        unsafety,
        abi,
        fn_token,
        ident,
        generics,
        paren_token: _paren_token,
        inputs,
        variadic,
        output,
    } = &func.sig;
    let generics_names = generics
        .params
        .iter()
        .filter_map(|x| match x {
            GenericParam::Type(t) => Some(t.ident.clone()),
            GenericParam::Const(c) => Some(c.ident.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let inner_generics = if generics_names.is_empty() {
        quote! {}
    } else {
        quote! { :: < #(#generics_names),* > }
    };
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let arg_names = inputs
        .iter()
        .map(|x| match x {
            FnArg::Typed(pat) => {
                let pat = &pat.pat;
                quote! { #pat }
            }
            FnArg::Receiver(r) => {
                quote! { #r }
            }
        })
        .collect::<Vec<_>>();

    let instances = levels
        .iter()
        .map(|level| {
            let level_ident = syn::Ident::new(&level, level_span);
            let features = features_for_level(&level);
            let tf_attr = if !features.is_empty() {
                let tf_list = features.join(",");
                quote! {
                    #[target_feature(enable = #tf_list)]
                }
            } else {
                quote! {}
            };
            let mod_ident = syn::Ident::new(&mod_for_level(&level), level_span);
            let mut edited_block = block.clone();
            let mut level_info = LevelInfo {
                level: level.clone(),
            };
            syn::visit_mut::visit_block_mut(&mut level_info, &mut edited_block);
            let instance_vis = if do_module {
                quote! { pub }
            } else {
                quote! {}
            };

            quote! {
                #tf_attr
                #[inline]
                #instance_vis #unsafety
                fn #level_ident #impl_generics (#inputs #variadic) #output #where_clause {
                    use fearless_simd::#mod_ident as simd;
                    #edited_block
                }
            }
        })
        .collect::<Vec<_>>();

    let mut dispatch = vec![];
    let mut do_panic = true;
    for level in &levels {
        let level_ident = syn::Ident::new(&level, level_span);
        let fn_path = if do_module {
            quote! { #ident::#level_ident }
        } else {
            quote! { #level_ident }
        };
        if let Some(arch) = arch_for_level(&level) {
            let det_str = detect_macro_for_arch(arch);
            let det_ident = syn::Ident::new(det_str, level_span);
            let f_checks = features_for_level(&level)
                .iter()
                .map(|x| {
                    quote! { std::arch::#det_ident!(#x) }
                })
                .collect::<Vec<_>>();
            dispatch.push(quote! {
                #[cfg(target_arch = #arch)]
                if #(#f_checks &&)* true {
                    // SAFETY: target features are checked by the if condition.
                    unsafe {
                        return #fn_path #inner_generics (#(#arg_names,)*)
                    }
                }
            });
        } else {
            dispatch.push(quote! {
                #fn_path #inner_generics (#(#arg_names,)*)
            });
            do_panic = false;
        }
    }
    if do_panic {
        dispatch.push(quote! {
            panic!("CPU does not support needed features, and no fallback provided");
        });
    }

    let result = if do_module {
        quote! {
            #vis mod #ident {
                use super::*;
                #(#instances)*
            }
            #(#attrs)* #vis #constness #unsafety #asyncness #abi
            #fn_token #ident #impl_generics (#inputs #variadic) #output #where_clause {
                #(#dispatch)*
            }
        }
    } else {
        quote! {
            #(#attrs)* #vis #constness #unsafety #asyncness #abi
            #fn_token #ident #impl_generics (#inputs #variadic) #output #where_clause {
                #(#instances)*
                #(#dispatch)*
            }
        }
    };
    result.into()
}

impl Parse for AttributeArg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(AttributeArg {
            key: input.parse()?,
            _equals: input.parse()?,
            value: input.parse()?,
        })
    }
}

impl Parse for AttributeValue {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(LitStr) {
            Ok(AttributeValue::String(input.parse()?))
        } else {
            Ok(AttributeValue::Bool(input.parse()?))
        }
    }
}

// Based on "Conditional compilation" section of Rust reference

#[derive(Debug)]
enum ConfigurationPredicate {
    Option(ConfigurationOption),
    All(ConfigurationAll),
    Any(ConfigurationAny),
    Not(ConfigurationNot),
}

#[derive(Debug)]
struct ConfigurationOption {
    ident: Ident,
    // Note: raw string literal is part of the syntax, but we just bail
    tail: Option<(Token![=], LitStr)>,
}

#[derive(Debug)]
struct ConfigurationAll {
    all_token: Ident,
    paren_token: Paren,
    list: Punctuated<ConfigurationPredicate, Token![,]>,
}

#[derive(Debug)]
struct ConfigurationAny {
    any_token: Ident,
    paren_token: Paren,
    list: Punctuated<ConfigurationPredicate, Token![,]>,
}

#[derive(Debug)]
struct ConfigurationNot {
    not_token: Ident,
    paren_token: Paren,
    negatee: Box<ConfigurationPredicate>,
}

impl Parse for ConfigurationPredicate {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        if input.peek(Token![=]) {
            let equals = input.parse()?;
            let literal = input.parse()?;
            Ok(ConfigurationPredicate::Option(ConfigurationOption {
                ident,
                tail: Some((equals, literal)),
            }))
        } else if ident == "all" {
            let content;
            let paren_token = parenthesized!(content in input);
            let list = content.parse_terminated(ConfigurationPredicate::parse, Token![,])?;
            Ok(ConfigurationPredicate::All(ConfigurationAll {
                all_token: ident,
                paren_token,
                list,
            }))
        } else if ident == "any" {
            let content;
            let paren_token = parenthesized!(content in input);
            let list = content.parse_terminated(ConfigurationPredicate::parse, Token![,])?;
            Ok(ConfigurationPredicate::Any(ConfigurationAny {
                any_token: ident,
                paren_token,
                list,
            }))
        } else if ident == "not" {
            let content;
            let paren_token = parenthesized!(content in input);
            let list = content.parse_terminated(ConfigurationPredicate::parse, Token![,])?;
            if list.len() != 1 {
                return Err(content.error("only 1 argument expected for cfg not()"));
            } else {
                let negatee = list.into_iter().next().unwrap();
                Ok(ConfigurationPredicate::Not(ConfigurationNot {
                    not_token: ident,
                    paren_token,
                    negatee: Box::new(negatee),
                }))
            }
        } else {
            Ok(ConfigurationPredicate::Option(ConfigurationOption {
                ident,
                tail: None,
            }))
        }
    }
}

impl ToTokens for ConfigurationPredicate {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            ConfigurationPredicate::Option(item) => {
                item.ident.to_tokens(tokens);
                if let Some((tok, lit)) = &item.tail {
                    tok.to_tokens(tokens);
                    lit.to_tokens(tokens);
                }
            }
            ConfigurationPredicate::All(item) => {
                item.all_token.to_tokens(tokens);
                item.paren_token.surround(tokens, |tokens| {
                    item.list.to_tokens(tokens);
                });
            }
            ConfigurationPredicate::Any(item) => {
                item.any_token.to_tokens(tokens);
                item.paren_token.surround(tokens, |tokens| {
                    item.list.to_tokens(tokens);
                });
            }
            ConfigurationPredicate::Not(item) => {
                item.not_token.to_tokens(tokens);
                item.paren_token.surround(tokens, |tokens| {
                    item.negatee.to_tokens(tokens);
                });
            }
        }
    }
}

struct LevelInfo {
    level: String,
}

impl VisitMut for LevelInfo {
    fn visit_attribute_mut(&mut self, i: &mut Attribute) {
        if i.style != AttrStyle::Outer {
            return;
        }
        if let Meta::List(l) = &mut i.meta {
            if l.path.is_ident("cfg") {
                if let Ok(mut cfg) = l.parse_args::<ConfigurationPredicate>() {
                    let span = l.tokens.span();
                    self.edit_cfg(&mut cfg, span);
                    l.tokens = cfg.into_token_stream();
                }
            }
        }
    }
}

impl LevelInfo {
    fn edit_cfg(&self, cfg: &mut ConfigurationPredicate, span: Span) {
        match cfg {
            ConfigurationPredicate::Option(item) => {
                if item.ident == "fearless_simd_level" {
                    if let Some((_, arg)) = &mut item.tail {
                        let value = self.supports_level(&arg.value());
                        Self::resolve_cfg(cfg, value, span);
                    }
                } else if item.ident == "target_feature" {
                    if let Some((_, arg)) = &mut item.tail {
                        if self.supports_target_feature(&arg.value()) {
                            // Note: we don't resolve false values, to support querying
                            // features in the compiler's cfg that we don't know about.
                            Self::resolve_cfg(cfg, true, span);
                        }
                    }
                }
            }
            ConfigurationPredicate::All(item) => {
                for child in &mut item.list {
                    self.edit_cfg(child, span);
                }
            }
            ConfigurationPredicate::Any(item) => {
                for child in &mut item.list {
                    self.edit_cfg(child, span);
                }
            }
            ConfigurationPredicate::Not(item) => self.edit_cfg(&mut item.negatee, span),
        }
    }

    fn supports_level(&self, level_query: &str) -> bool {
        if self.level == level_query {
            true
        } else {
            matches!((self.level.as_str(), level_query), ("fp16", "neon"))
        }
    }

    // TODO: say yes for implied features, see
    // https://github.com/rust-lang/rust/blob/52890e82153cd8716d97a96f47fb6ac99dec65be/compiler/rustc_target/src/target_features.rs#L207
    fn supports_target_feature(&self, tf_query: &str) -> bool {
        features_for_level(&self.level)
            .iter()
            .any(|tf| *tf == tf_query)
    }

    fn resolve_cfg(cfg: &mut ConfigurationPredicate, value: bool, span: Span) {
        if value {
            *cfg = ConfigurationPredicate::All(ConfigurationAll {
                all_token: Ident::new("all", span),
                paren_token: Paren::default(),
                list: Punctuated::new(),
            });
        } else {
            *cfg = ConfigurationPredicate::Any(ConfigurationAny {
                any_token: Ident::new("any", span),
                paren_token: Paren::default(),
                list: Punctuated::new(),
            });
        }
    }
}
