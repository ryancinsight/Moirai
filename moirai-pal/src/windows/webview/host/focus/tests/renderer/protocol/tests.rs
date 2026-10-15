//! Analytical sparse-index examples from the CDP representation contract.

use super::*;
use serde_json::json;

fn example() -> Value {
    json!({"strings":["https://metis.invalid/index.html","frame-a","INPUT","id",
        "reference","weight","concentration","dose","患者-🦀-three","96","8","0.125"],
        "documents":[{"documentURL":0,"frameId":1,"nodes":{
            "nodeType":[1,1,1,1],"nodeName":[2,2,2,2],"backendNodeId":[31,32,33,34],
            "attributes":[[3,4],[3,5],[3,6],[3,7]],
            "inputValue":{"index":[0,1,2,3],"value":[8,9,10,11]}}}]})
}

#[test]
fn sparse_live_values_preserve_unicode_and_do_not_read_attributes() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let value = tracing::info_span!("arrange").in_scope(example);
    let snapshot = tracing::info_span!("act")
        .in_scope(|| decode(&value.to_string()).expect("analytical sparse input values"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(snapshot.frame, "frame-a");
        for (id, text, backend_node) in [
            ("reference", "患者-🦀-three", 31),
            ("weight", "96", 32),
            ("concentration", "8", 33),
            ("dose", "0.125", 34),
        ] {
            let field = &snapshot.fields[id];
            assert_eq!(field.backend_node, backend_node);
            assert_eq!(field.units, text.encode_utf16().collect::<Vec<_>>());
        }
    });
}

#[test]
fn explicit_empty_is_distinct_from_missing_sparse_entry() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let mut value = tracing::info_span!("arrange").in_scope(example);
    value["documents"][0]["nodes"]["inputValue"]["value"][0] = (-1).into();
    let snapshot = tracing::info_span!("act")
        .in_scope(|| decode(&value.to_string()).expect("analytical sparse input values"));
    tracing::info_span!("assert")
        .in_scope(|| assert!(snapshot.fields["reference"].units.is_empty()));
    value["documents"][0]["nodes"]["inputValue"]["index"]
        .as_array_mut()
        .expect("analytical array column")
        .remove(0);
    value["documents"][0]["nodes"]["inputValue"]["value"]
        .as_array_mut()
        .expect("analytical array column")
        .remove(0);
    assert!(decode(&value.to_string()).is_err());
}

#[test]
fn malformed_indexes_columns_documents_and_field_ownership_reject() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    for (pointer, replacement) in [
        ("/strings/3", json!(false)),
        ("/documents/0/frameId", json!(-1)),
        ("/documents/0/nodes/nodeName/0", json!(99)),
        ("/documents/0/nodes/nodeType/0", json!(3)),
        ("/documents/0/nodes/backendNodeId/0", json!(0)),
        ("/documents/0/nodes/backendNodeId/1", json!(31)),
        ("/documents/0/nodes/attributes/0", json!([3])),
        ("/documents/0/nodes/attributes/0", json!([3, 4, 3, 4])),
        ("/documents/0/nodes/attributes/1", json!([3, 4])),
        ("/documents/0/nodes/inputValue/index/1", json!(0)),
        ("/documents/0/nodes/inputValue/index/1", json!(4)),
        ("/documents/0/nodes/inputValue/value/1", json!(-2)),
        ("/documents/0/nodes/inputValue/value/1", json!(99)),
        ("/documents/0/nodes/inputValue/value", json!([8])),
        ("/documents/0/nodes/nodeName", json!([2])),
    ] {
        let _case = tracing::info_span!("snapshot_rejection", pointer).entered();
        let mut value = tracing::info_span!("arrange").in_scope(example);
        *value.pointer_mut(pointer).expect("declared mutation path") = replacement;
        let result = tracing::info_span!("act").in_scope(|| decode(&value.to_string()));
        tracing::info_span!("assert").in_scope(|| assert!(result.is_err(), "{pointer}: {value}"));
    }
    let mut value = example();
    let document = value["documents"][0].clone();
    value["documents"]
        .as_array_mut()
        .expect("analytical document array")
        .push(document);
    assert!(decode(&value.to_string()).is_err());
}
