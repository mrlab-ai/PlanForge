//! `#[derive(ApplyOptions)]` proc macro for `planforge-search` typed configs.
//!
//! Emits `impl ApplyOptions for X` where `X` is a struct with named fields.
//! Each non-`#[option(skip)]` field becomes a match arm in the generated
//! `apply_options(&mut self, args: &[ConfigArg]) -> Result<(), String>`.
//!
//! Field attributes:
//! - `#[option(skip)]`            — field is not exposed as a CLI option.
//! - `#[option(rename = "alias")]`— CLI key differs from field name.
//! - `#[option(flatten)]`         — catch-all: unknown keys delegate to this
//!                                  field's `apply_options` (the field's
//!                                  type must also implement `ApplyOptions`).
//!                                  Only one `flatten` per struct.
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

        if attrs.flatten {
            if flatten_field.is_some() {
                return Err(Error::new_spanned(
                    field,
                    "only one field may use `#[option(flatten)]`",
                ));
            }
            flatten_field = Some(field_ident);
            continue;
        }

        let key = attrs.rename.unwrap_or_else(|| field_ident.to_string());
        order_keys.push(key.clone());

        let key_lit = syn::LitStr::new(&key, field.span());
        let arm = quote! {
            #key_lit => {
                self.#field_ident = ::planforge_search::config::FromOptionValue::from_option_value(value)?;
            }
        };
        arms.push(arm);
    }

    let catchall = if let Some(field) = flatten_field {
        quote! {
            other => {
                // Reconstruct a single ConfigArg and delegate to the nested
                // ApplyOptions impl.
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
                    "expected `#[option(skip)]`, `#[option(flatten)]`, or `#[option(rename = \"…\")]`",
                ));
            }
        };
        let nested =
            list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for meta in nested {
            match &meta {
                Meta::Path(path) if path.is_ident("skip") => out.skip = true,
                Meta::Path(path) if path.is_ident("flatten") => out.flatten = true,
                Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                    let value = match &nv.value {
                        syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) => s.value(),
                        _ => {
                            return Err(Error::new_spanned(
                                &nv.value,
                                "`rename` value must be a string literal",
                            ));
                        }
                    };
                    out.rename = Some(value);
                }
                other => {
                    return Err(Error::new_spanned(
                        other,
                        "unknown `#[option(...)]` attribute (supported: skip, flatten, rename)",
                    ));
                }
            }
        }
    }
    Ok(out)
}

// Small extension trait so we can call `.span()` on a Field.
trait Spanned {
    fn span(&self) -> proc_macro2::Span;
}

impl Spanned for Field {
    fn span(&self) -> proc_macro2::Span {
        syn::spanned::Spanned::span(self)
    }
}
