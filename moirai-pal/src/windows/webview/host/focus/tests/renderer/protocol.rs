//! Exact sparse DOMSnapshot decoding, independent of the accessibility tree.

use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
};

pub(super) const FIELDS: [&str; 4] = ["reference", "weight", "concentration", "dose"];

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
pub(super) struct Field {
    node: usize,
    backend_node: u64,
    units: Vec<u16>,
}

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
pub(super) struct Snapshot {
    frame: String,
    fields: BTreeMap<String, Field>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn array(value: &Value) -> io::Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("DOMSnapshot array missing"))
}

fn index(value: &Value) -> io::Result<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("DOMSnapshot index is not representable"))
}

fn string<'a>(strings: &'a [Value], value: &Value) -> io::Result<&'a str> {
    strings
        .get(index(value)?)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("DOMSnapshot string index is outside the string table"))
}

/// Sparse absence does not mean an empty input. Blink emits an explicit -1
/// string index for the empty live value; every other index must resolve.
pub(super) fn decode(encoded: &str) -> io::Result<Snapshot> {
    let reply: Value = serde_json::from_str(encoded).map_err(io::Error::other)?;
    let strings = array(&reply["strings"])?;
    if strings.iter().any(|v| !v.is_string()) {
        return Err(invalid("DOMSnapshot string table contains a non-string"));
    }
    let mut documents = array(&reply["documents"])?.iter().filter_map(|document| {
        match string(strings, &document["documentURL"]) {
            Ok("https://metis.invalid/index.html") => Some(Ok(document)),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        }
    });
    let document = documents
        .next()
        .ok_or_else(|| invalid("DOMSnapshot selected document missing"))??;
    if documents.next().is_some() {
        return Err(invalid("DOMSnapshot selected document is ambiguous"));
    }
    let frame = string(strings, &document["frameId"])?;
    if frame.is_empty() {
        return Err(invalid("DOMSnapshot frame identity missing"));
    }
    let nodes = &document["nodes"];
    let kinds = array(&nodes["nodeType"])?;
    let names = array(&nodes["nodeName"])?;
    let backend = array(&nodes["backendNodeId"])?;
    let attributes = array(&nodes["attributes"])?;
    let count = kinds.len();
    if [names.len(), backend.len(), attributes.len()]
        .into_iter()
        .any(|length| length != count)
    {
        return Err(invalid("DOMSnapshot node columns disagree"));
    }
    let sparse = &nodes["inputValue"];
    let indexes = array(&sparse["index"])?;
    let values = array(&sparse["value"])?;
    if indexes.len() != values.len() {
        return Err(invalid("DOMSnapshot inputValue columns disagree"));
    }
    let mut live = BTreeMap::new();
    for (node, value) in indexes.iter().zip(values) {
        let node = index(node)?;
        if node >= count || live.contains_key(&node) {
            return Err(invalid(
                "DOMSnapshot inputValue node is duplicate or absent",
            ));
        }
        let text = if value.as_i64() == Some(-1) {
            ""
        } else {
            string(strings, value)?
        };
        live.insert(node, text);
    }
    let mut fields = BTreeMap::new();
    let mut owners = BTreeSet::new();
    for (node, attributes) in attributes.iter().enumerate() {
        let attributes = array(attributes)?;
        if attributes.len() % 2 != 0 {
            return Err(invalid("DOMSnapshot attributes are not name/value pairs"));
        }
        let mut identifier = None;
        for pair in attributes.chunks_exact(2) {
            let name = string(strings, &pair[0])?;
            let value = string(strings, &pair[1])?;
            if name == "id" && identifier.replace(value).is_some() {
                return Err(invalid("DOMSnapshot node has duplicate id attributes"));
            }
        }
        let Some(id) = identifier.filter(|id| FIELDS.contains(id)) else {
            continue;
        };
        if kinds[node].as_u64() != Some(1) || string(strings, &names[node])? != "INPUT" {
            return Err(invalid("DOMSnapshot field is not an INPUT element"));
        }
        let backend_node = backend[node]
            .as_u64()
            .filter(|id| *id > 0)
            .ok_or_else(|| invalid("DOMSnapshot backend node identity missing"))?;
        if !owners.insert(backend_node) {
            return Err(invalid("DOMSnapshot fields share a backend node identity"));
        }
        let text = live
            .get(&node)
            .ok_or_else(|| invalid("DOMSnapshot field has no explicit live inputValue"))?;
        let field = Field {
            node,
            backend_node,
            units: text.encode_utf16().collect(),
        };
        if fields.insert(id.to_owned(), field).is_some() {
            return Err(invalid("DOMSnapshot field identity is ambiguous"));
        }
    }
    if fields.len() != FIELDS.len() {
        return Err(invalid("DOMSnapshot does not contain all four live fields"));
    }
    Ok(Snapshot {
        frame: frame.to_owned(),
        fields,
    })
}

#[cfg(test)]
mod tests;
