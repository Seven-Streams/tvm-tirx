/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! The procedural macros of `tvm-tirx`, which re-exports them: use them as `tvm_tirx::tirx_type`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Error, Fields, LitStr, Meta, Path};
use syn::{Result, Visibility};

/// The reflection flags a field takes as an attribute of the same name: `#[seq_hash_ignore]` is
/// tvm-ffi's `FieldFlags::SEQ_HASH_IGNORE`.
const FIELD_FLAGS: &[&str] = &[
    "writable",
    "seq_hash_ignore",
    "seq_hash_def_pattern",
    "seq_hash_def_simple",
    "repr_off",
    "compare_off",
    "hash_off",
    "init_off",
    "kw_only",
];

/// Define an IR type in Rust, as C++ defines `Name` and `NameNode` for an `ir.Type` such as
/// `PrimType` or `TupleType`: a type of expressions, registered under the type key with the
/// reflection of every field, so that the C++ structural equality and hashing, deep copy, `repr`,
/// the TVMScript printer and Python see the fields.
///
/// ```ignore
/// use tvm_tirx::ir::{PrimType, Type, Var, UNKNOWN_LOC};
/// use tvm_tirx::tirx_type;
///
/// /// The type of a pair of values.
/// #[tirx_type("my_ext.PairType")]
/// pub struct PairType {
///     pub first: Type,
///     pub second: Type,
/// }
///
/// impl PairType {
///     pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
///         Self::from_complete_fields(UNKNOWN_LOC, first, second)
///     }
/// }
///
/// let pair = PairType::new(PrimType::int(32), PrimType::float(32));
/// let p = Var::new("p", pair, UNKNOWN_LOC);
/// ```
///
/// - `NameObj` is the node: the fields of `ir.Type`, then the struct's, each reflected and with
///   its visibility. A field takes reflection flags as attributes, such as `#[seq_hash_ignore]`.
/// - `Name` is the reference, with the struct's attributes and visibility. It dereferences to
///   `NameObj`, and that to `TypeObj`, so `name.field` and `name.loc` read fields; it converts
///   into `Type`, and a `NameObj` converts into it.
/// - `Name::from_complete_fields(loc, ...)` allocates one from all its fields in order: the
///   location, which `ir.Type` holds, then the struct's, as the generated bindings of the C++ types
///   take them. A constructor with the type's meaning is written on top of it.
///
/// The type is final. Define it at module level: the definitions are in a module of their own
/// that sees the enclosing module's names, and are re-exported from it.
#[proc_macro_attribute]
pub fn tirx_type(args: TokenStream, item: TokenStream) -> TokenStream {
    let key = parse_macro_input!(args as LitStr);
    let item = parse_macro_input!(item as DeriveInput);
    expand(key, item).unwrap_or_else(|error| error.to_compile_error()).into()
}

fn expand(key: LitStr, item: DeriveInput) -> Result<TokenStream2> {
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(Error::new_spanned(&item.generics, "a type cannot be generic"));
    }
    let Data::Struct(data) = &item.data else {
        return Err(Error::new(item.ident.span(), "expected a struct"));
    };
    let named = match &data.fields {
        Fields::Named(fields) => fields.named.iter().collect(),
        Fields::Unit => Vec::new(),
        Fields::Unnamed(fields) => {
            return Err(Error::new_spanned(fields, "expected named fields"));
        }
    };

    let mut fields = Vec::new();
    let mut field_names = Vec::new();
    let mut field_types = Vec::new();
    for field in named {
        let name = field.ident.clone().expect("named field");
        if name == "base" || name == "loc" {
            let message = format!("`{name}` is a field of `ir.Type` or a constructor argument");
            return Err(Error::new(name.span(), message));
        }
        let mut flags = Vec::new();
        let mut attrs = Vec::new();
        for attr in &field.attrs {
            let flag = attr.path.get_ident().filter(|id| FIELD_FLAGS.iter().any(|f| *id == f));
            if let Some(flag) = flag {
                if !matches!(attr.parse_meta()?, Meta::Path(_)) {
                    return Err(Error::new_spanned(attr, "a flag takes no arguments"));
                }
                flags.push(flag.clone());
            } else if attr.path.is_ident("reflect") {
                let message = "every field is reflected; write its flags as attributes, such as \
                               `#[seq_hash_ignore]`";
                return Err(Error::new_spanned(attr, message));
            } else {
                attrs.push(attr);
            }
        }
        let reflect =
            if flags.is_empty() { quote!(#[reflect]) } else { quote!(#[reflect(#(#flags),*)]) };
        let vis = inner_visibility(&field.vis);
        let ty = &field.ty;
        fields.push(quote!(#(#attrs)* #reflect #vis #name: #ty,));
        field_names.push(name);
        field_types.push(ty);
    }

    let ir = quote!(::tvm_tirx::ir);
    let into = quote!(::std::convert::Into);
    let attrs = &item.attrs;
    let vis = &item.vis;
    let name = &item.ident;
    let obj = format_ident!("{}Obj", name);
    let module = format_ident!("__tirx_type_{}", snake_case(&name.to_string()));
    let obj_doc = format!("The node of [`{name}`].");
    Ok(quote! {
        // The derives name `tvm_ffi`, which a crate that depends only on `tvm-tirx` does not have:
        // the definitions get it in a module of their own and are re-exported from it.
        #[doc(hidden)]
        mod #module {
            use super::*;
            use ::tvm_tirx::tvm_ffi;

            #[doc = #obj_doc]
            #[repr(C)]
            #[derive(tvm_ffi::derive::Object)]
            #[type_key = #key]
            #[type_register]
            #[type_final]
            pub struct #obj {
                base: #ir::TypeObj,
                #(#fields)*
            }

            #(#attrs)*
            #[repr(C)]
            #[derive(tvm_ffi::derive::ObjectRef, Clone)]
            pub struct #name {
                base: tvm_ffi::ObjectArc<#obj>,
            }

            impl ::std::ops::Deref for #name {
                type Target = #obj;
                fn deref(&self) -> &#obj {
                    &self.base
                }
            }

            impl ::std::ops::Deref for #obj {
                type Target = #ir::TypeObj;
                fn deref(&self) -> &#ir::TypeObj {
                    &self.base
                }
            }

            impl ::std::convert::From<#obj> for #name {
                fn from(node: #obj) -> Self {
                    Self { base: tvm_ffi::ObjectArc::new(node) }
                }
            }

            impl #name {
                /// Lossless complete-field allocation.
                #[allow(clippy::too_many_arguments)]
                pub fn from_complete_fields(
                    loc: impl #into<#ir::Location>,
                    #(#field_names: impl #into<#field_types>,)*
                ) -> Self {
                    let base = ::tvm_tirx::ty::type_base(loc.into());
                    Self::from(#obj { base, #(#field_names: #field_names.into()),* })
                }
            }

            tvm_ffi::impl_object_upcast!(#name => #ir::Type);
        }

        // A type used only through its reference, or the other way round, is not an unused import.
        #[allow(unused_imports)]
        #vis use #module::{#name, #obj};
    })
}

/// `vis` as seen from the module that holds the definitions, a child of the one written in.
fn inner_visibility(vis: &Visibility) -> TokenStream2 {
    match vis {
        Visibility::Public(_) | Visibility::Crate(_) => quote!(#vis),
        Visibility::Inherited => quote!(pub(super)),
        Visibility::Restricted(restricted) => {
            let path: &Path = &restricted.path;
            match path.segments.first().map(|segment| segment.ident.to_string()).as_deref() {
                Some("crate") => quote!(#vis),
                Some("self") => {
                    let rest = path.segments.iter().skip(1);
                    quote!(pub(in super #(::#rest)*))
                }
                _ => quote!(pub(in super::#path)),
            }
        }
    }
}

/// `TaggedExpr` as `tagged_expr`.
fn snake_case(name: &str) -> String {
    let mut snake = String::new();
    for (index, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if index > 0 {
                snake.push('_');
            }
            snake.extend(c.to_lowercase());
        } else {
            snake.push(c);
        }
    }
    snake
}
