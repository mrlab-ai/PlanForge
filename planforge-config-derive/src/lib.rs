//! `#[derive(ApplyOptions)]` proc macro for `planforge-search` typed configs.
//!
//! This is configuration support for the search end of PlanForge's PDDL
//! translation → SAS+ task → search pipeline. It generates option assignment
//! code for typed search and heuristic configurations; it does not translate
//! tasks or run search itself.
//!
//! Positional arguments are explicit field annotations, so reordering other
//! fields cannot silently change the accepted CLI syntax.
//!
//! Emits `impl ApplyOptions for X` where `X` is a struct with named fields.
//! Each non-`#[option(skip)]` field becomes a match arm in the generated
//! `apply_options(&mut self, args: &[ConfigArg]) -> Result<(), String>`.
//!
//! Field attributes:
//! - `#[option(positional)]` — accepts the field in positional order.
//! - `#[option(skip)]` — not exposed as a CLI option.
//! - `#[option(rename = "alias")]` — CLI key differs from field name.
//! - `#[option(flatten)]` — catch-all: unknown keys delegate to this field's
//!   `apply_options` (the field's type must impl `ApplyOptions`). Only one
//!   `flatten` per struct, and a struct with one takes named options only.
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
            if attrs.rename.is_some() {
                return Err(Error::new_spanned(
                    field,
                    "`rename` cannot be combined with `flatten`",
                ));
            }
            if flatten_field.is_some() {
                return Err(Error::new_spanned(
                    field,
                    "only one field may use `#[option(flatten)]`",
                ));
            }
            flatten_field = Some(field_ident);
            continue;
        }

        // Plain assignment field.
        let key = attrs.rename.unwrap_or_else(|| field_ident.to_string());
        if attrs.positional {
            order_keys.push(key.clone());
        }
        let key_lit = syn::LitStr::new(&key, field_span(field));

        arms.push(quote! {
            #key_lit => {
                self.#field_ident =
                    ::planforge_search::config::FromOptionValue::from_option_value(value)?;
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
            other => return Err(format!("unknown option `{other}`")),
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
    positional: bool,
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
        let nested = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
        for meta in nested {
            match &meta {
                Meta::Path(path) if path.is_ident("positional") => out.positional = true,
                Meta::Path(path) if path.is_ident("skip") => out.skip = true,
                Meta::Path(path) if path.is_ident("flatten") => out.flatten = true,
                Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                    out.rename = Some(string_value(nv)?);
                }
                other => {
                    return Err(Error::new_spanned(
                        other,
                        "unknown `#[option(...)]` attribute (supported: positional, skip, flatten, \
                         rename)",
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn string_value(nv: &syn::MetaNameValue) -> Result<String, Error> {
    match &nv.value {
        syn::Expr::Lit(syn::ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        _ => Err(Error::new_spanned(
            &nv.value,
            "value must be a string literal",
        )),
    }
}

fn field_span(field: &Field) -> proc_macro2::Span {
    syn::spanned::Spanned::span(field)
}

/// Decode a data-carrying option enum, with one named call per variant.
#[proc_macro_derive(FromOptionValue, attributes(option))]
pub fn derive_from_option_value(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match build_enum_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn build_enum_impl(input: &DeriveInput) -> Result<TokenStream2, Error> {
    let Data::Enum(data) = &input.data else {
        return Err(Error::new_spanned(
            input,
            "FromOptionValue requires an enum",
        ));
    };
    let ty = &input.ident;
    let mut arms = Vec::new();
    let mut names = Vec::new();
    for variant in &data.variants {
        let mut name = None;
        let mut aliases = Vec::new();
        for attr in &variant.attrs {
            if !attr.path().is_ident("option") {
                continue;
            }
            for meta in attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)? {
                match meta {
                    Meta::NameValue(nv) if nv.path.is_ident("rename") => {
                        if name.is_some() {
                            return Err(Error::new_spanned(nv, "duplicate call name"));
                        }
                        name = Some(string_value(&nv)?);
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("alias") => {
                        aliases.push(string_value(&nv)?)
                    }
                    other => return Err(Error::new_spanned(other, "expected rename or alias")),
                }
            }
        }
        let name = name.ok_or_else(|| {
            Error::new_spanned(
                variant,
                "each variant needs #[option(rename = \"call_name\")]",
            )
        })?;
        names.push(name.clone());
        let ident = &variant.ident;
        let body = match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let config = &fields.unnamed[0].ty;
                quote! { Ok(Self::#ident(::planforge_search::config::configure(
                    name, <#config as ::std::default::Default>::default(), args,
                )?)) }
            }
            Fields::Unit => quote! {
                if args.is_empty() { Ok(Self::#ident) }
                else { Err(format!("`{name}` does not accept arguments")) }
            },
            _ => {
                return Err(Error::new_spanned(
                    variant,
                    "expected a unit variant or one configuration field",
                ));
            }
        };
        arms.push(quote! { #name #( | #aliases )* => { #body } });
    }
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::planforge_search::config::FromOptionValue for #ty #type_generics #where_clause {
            fn from_option_value(value: &::planforge_search::config::ConfigValue) -> Result<Self, String> {
                let (name, args) = match value {
                    ::planforge_search::config::ConfigValue::Atom(name) => (name.as_str(), &[][..]),
                    ::planforge_search::config::ConfigValue::Call(call) => (call.name(), call.args()),
                    ::planforge_search::config::ConfigValue::List(_) => return Err(format!("expected {}, got a list", stringify!(#ty))),
                };
                match name.to_ascii_lowercase().as_str() {
                    #(#arms,)*
                    _ => Err(format!("unknown {} `{name}`; expected one of {}", stringify!(#ty), [#(#names),*].join(", "))),
                }
            }
        }
    })
}
