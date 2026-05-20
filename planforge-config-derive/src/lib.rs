//! `#[derive(ApplyOptions)]` proc macro for `planforge-search` typed configs.
//!
//! Emits `impl ApplyOptions for X` where `X` is a struct with named fields.
//! Each non-`#[option(skip)]` field becomes a match arm in the generated
//! `apply_options(&mut self, args: &[ConfigArg]) -> Result<(), String>`.
//!
//! Field attributes:
//! - `#[option(skip)]`               — not exposed as a CLI option.
//! - `#[option(rename = "alias")]`   — CLI key differs from field name.
//! - `#[option(flatten)]`            — catch-all: unknown keys delegate to
//!                                     this field's `apply_options` (the
//!                                     field's type must impl `ApplyOptions`).
//!                                     Only one `flatten` per struct.
//! - `#[option(nested = "key")]`     — explicit named arm whose value is a
//!                                     `Call`; its args are routed through
//!                                     this field's `apply_options`. Can be
//!                                     combined with `flatten` on the same
//!                                     field (so the same nested config is
//!                                     reachable both ways).
//! - `#[option(also_sets = "path")]` — after setting the field, also assign
//!                                     `self.<path>` from it. Useful for
//!                                     coupled keys mirrored into a sub-
//!                                     config. Repeatable: multiple paths
//!                                     write multiple times.
//!
//! `flatten` and `nested` imply a struct-shaped field (delegation target);
//! `also_sets` is for plain assignment-style fields. `also_sets` and
//! `flatten`/`nested` are mutually exclusive — the derive errors if both
//! are set on the same field.
//!
//! The trait, primitive `FromOptionValue` impls, and `for_each_option`
//! helper live in `planforge_search::config`. The emitted code references
//! them via absolute `::planforge_search::config::…` paths.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Data, DeriveInput, Error, Field, Fields, Lit, Meta, MetaList, Token, parse_macro_input,
    punctuated::Punctuated,
};

#[proc_macro_derive(ApplyOptions, attributes(option))]
pub fn derive_apply_options(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match build_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn build_impl(input: &DeriveInput) -> Result<TokenStream2, Error> {
    let ty = &input.ident;
    let ty_str = ty.to_string();

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(Error::new_spanned(
                    ty,
                    "ApplyOptions requires a struct with named fields",
                ));
            }
        },
        _ => {
            return Err(Error::new_spanned(
                ty,
                "ApplyOptions can only be derived on structs",
            ));
        }
    };

    let mut order_keys: Vec<String> = Vec::new();
    let mut arms: Vec<TokenStream2> = Vec::new();
    let mut flatten_field: Option<&syn::Ident> = None;

    for field in fields.iter() {
        let attrs = parse_field_attrs(field)?;
        let field_ident = field.ident.as_ref().expect("named field");

        if attrs.skip {
            continue;
        }

        let is_struct_field = attrs.flatten || attrs.nested.is_some();
        if is_struct_field && !attrs.also_sets.is_empty() {
            return Err(Error::new_spanned(
                field,
                "`also_sets` is for primitive/enum fields; \
                 combine `flatten`/`nested` with another field instead",
            ));
        }
        if is_struct_field && attrs.rename.is_some() {
            return Err(Error::new_spanned(
                field,
                "`rename` is for primitive/enum fields; use `nested = \"key\"` \
                 on struct-shaped fields to set the explicit key",
            ));
        }

        // Struct-shaped field: emit nested-arm and/or flatten catch-all.
        if is_struct_field {
            if let Some(nested_key) = &attrs.nested {
                order_keys.push(nested_key.clone());
                let key_lit = syn::LitStr::new(nested_key, field_span(field));
                arms.push(quote! {
                    #key_lit => {
                        let call = ::planforge_search::config::ConfigValue::as_call(value)?;
                        ::planforge_search::config::ApplyOptions::apply_options(
                            &mut self.#field_ident,
                            call.args(),
                        )?;
                    }
                });
            }
            if attrs.flatten {
                if flatten_field.is_some() {
                    return Err(Error::new_spanned(
                        field,
                        "only one field may use `#[option(flatten)]`",
                    ));
                }
                flatten_field = Some(field_ident);
            }
            continue;
        }

        // Plain assignment field.
        let key = attrs.rename.unwrap_or_else(|| field_ident.to_string());
        order_keys.push(key.clone());
        let key_lit = syn::LitStr::new(&key, field_span(field));

        // For each `also_sets = "path"`, emit `self.<path> = self.<field>.clone();`.
        let also_sets_tokens: Vec<TokenStream2> = attrs
            .also_sets
            .iter()
            .map(|path| {
                let path_expr: syn::Expr = syn::parse_str(&format!("self.{path}"))
                    .map_err(|e| {
                        Error::new_spanned(
                            field,
                            format!("invalid `also_sets` path `{path}`: {e}"),
                        )
                    })?;
                Ok(quote! { #path_expr = ::std::clone::Clone::clone(&self.#field_ident); })
            })
            .collect::<Result<_, Error>>()?;

        arms.push(quote! {
            #key_lit => {
                self.#field_ident =
                    ::planforge_search::config::FromOptionValue::from_option_value(value)?;
                #(#also_sets_tokens)*
            }
        });
    }

    let catchall = if let Some(field) = flatten_field {
        quote! {
            other => {
                let arg = ::planforge_search::config::ConfigArg::new(
                    Some(other.to_string()),
                    value.clone(),
                );
                ::planforge_search::config::ApplyOptions::apply_options(
                    &mut self.#field,
                    std::slice::from_ref(&arg),
                )?;
            }
        }
    } else {
        quote! {
            other => return Err(format!("unknown option `{other}` for `{}`", #ty_str)),
        }
    };

    let order_lits: Vec<syn::LitStr> = order_keys
        .iter()
        .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
        .collect();

    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics ::planforge_search::config::ApplyOptions for #ty #type_generics #where_clause {
            fn apply_options(
                &mut self,
                args: &[::planforge_search::config::ConfigArg],
            ) -> ::std::result::Result<(), ::std::string::String> {
                const ORDER: &[&str] = &[ #(#order_lits),* ];
                ::planforge_search::config::for_each_option(args, ORDER, |key, value| {
                    match key {
                        #(#arms)*
                        #catchall
                    }
                    Ok(())
                })
            }
        }
    })
}

#[derive(Default)]
struct FieldAttrs {
    skip: bool,
    flatten: bool,
    rename: Option<String>,
    nested: Option<String>,
    also_sets: Vec<String>,
}

fn parse_field_attrs(field: &Field) -> Result<FieldAttrs, Error> {
    let mut out = FieldAttrs::default();
    for attr in &field.attrs {
        if !attr.path().is_ident("option") {
            continue;
        }
        let list: MetaList = match &attr.meta {
            Meta::List(list) => list.clone(),
            _ => {
                return Err(Error::new_spanned(
                    attr,
                    "expected `#[option(skip)]`, `#[option(flatten)]`, \
                     `#[option(rename = \"…\")]`, `#[option(nested = \"…\")]`, \
                     or `#[option(also_sets = \"…\")]`",
                ));
            }
        };
        let nested = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for meta in nested {
            match &meta {
                Meta::Path(path) if path.is_ident("skip") => out.skip = true,
                Meta::Path(path) if path.is_ident("flatten") => out.flatten = true,
                Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                    out.rename = Some(string_value(nv)?);
                }
                Meta::NameValue(nv) if nv.path.is_ident("nested") => {
                    out.nested = Some(string_value(nv)?);
                }
                Meta::NameValue(nv) if nv.path.is_ident("also_sets") => {
                    out.also_sets.push(string_value(nv)?);
                }
                other => {
                    return Err(Error::new_spanned(
                        other,
                        "unknown `#[option(...)]` attribute (supported: skip, flatten, \
                         rename, nested, also_sets)",
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn string_value(nv: &syn::MetaNameValue) -> Result<String, Error> {
    match &nv.value {
        syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) => Ok(s.value()),
        _ => Err(Error::new_spanned(
            &nv.value,
            "value must be a string literal",
        )),
    }
}

fn field_span(field: &Field) -> proc_macro2::Span {
    syn::spanned::Spanned::span(field)
}
