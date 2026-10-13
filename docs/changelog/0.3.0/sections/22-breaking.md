### Breaking
- Removed the unused exported `thread_local_static!` platform macro. Internal
  runtime TLS now uses concrete std TLS or Melinoe `thread_cached!` sites.
- Removed the placeholder public distributed facade methods `Moirai::spawn_remote`, `Moirai::get_nodes`, `Moirai::register_node`, `MoiraiBuilder::enable_distributed`, and `MoiraiBuilder::node_id`; distributed iterator helper coverage remains in `moirai-iter`.
- Removed public type-suffixed `moirai_utils::simd` vector functions in favor of generic `add`, `mul`, `dot`, `sum`, `mean`, `variance`, and `matrix_mul_square<T, const N>` operations over sealed scalar traits.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
