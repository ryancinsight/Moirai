//! HTTP redirect policy and URI-reference resolution.
//!
//! Method changes follow [RFC 9110 section 15.4], and reference resolution
//! follows the algorithm and examples in [RFC 3986 sections 5.2 and 5.4].
//!
//! [RFC 9110 section 15.4]: https://www.rfc-editor.org/rfc/rfc9110.html#section-15.4
//! [RFC 3986 sections 5.2 and 5.4]: https://www.rfc-editor.org/rfc/rfc3986.html#section-5.2

use std::io;
use std::str::FromStr;

use crate::Origin;

/// Return whether `status` is an automatically followable redirect.
pub(super) const fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

/// Return whether redirect compatibility semantics replace the request with GET.
pub(super) fn redirects_to_get(status: u16, method: &str) -> bool {
    (matches!(status, 301 | 302) && method.eq_ignore_ascii_case("POST"))
        || (status == 303 && !method.eq_ignore_ascii_case("HEAD"))
}

/// Filter request fields before forwarding them to a redirect destination.
pub(super) fn forwarded_headers<'a>(
    headers: &[(&'a str, &'a str)],
    cross_origin: bool,
    body_dropped: bool,
) -> Vec<(&'a str, &'a str)> {
    headers
        .iter()
        .copied()
        .filter(|(name, _)| {
            let hop_by_hop = is_hop_by_hop(name, headers);
            let representation = body_dropped
                && (name
                    .get(..8)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("content-"))
                    || name.eq_ignore_ascii_case("expect"));
            let credential = cross_origin
                && (name.eq_ignore_ascii_case("authorization")
                    || name.eq_ignore_ascii_case("cookie")
                    || name.eq_ignore_ascii_case("cookie2"));
            !hop_by_hop && !representation && !credential
        })
        .collect()
}

fn is_hop_by_hop(name: &str, headers: &[(&str, &str)]) -> bool {
    name.eq_ignore_ascii_case("host")
        || name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("proxy-connection")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailer")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || headers.iter().any(|(header, value)| {
            header.eq_ignore_ascii_case("connection")
                && value
                    .split(',')
                    .any(|token| token.trim().eq_ignore_ascii_case(name))
        })
}

/// Resolve a redirect `Location` and require an absolute HTTP(S) result.
pub(super) fn resolve_redirect(base: &str, location: &str) -> io::Result<String> {
    let location = location.trim();
    if location.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "redirect Location is empty",
        ));
    }
    let resolved = resolve_reference(base, location)?;
    let without_fragment = resolved
        .split_once('#')
        .map_or(resolved.as_str(), |(target, _)| target)
        .to_owned();
    parse_url(&without_fragment).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid redirect Location {location:?}: {error}"),
        )
    })?;
    Ok(without_fragment)
}

/// Parse an absolute HTTP(S) URL into its pool key and request target.
pub(super) fn parse_url(url: &str) -> io::Result<(Origin, String)> {
    let uri = http::Uri::from_str(url).map_err(|error| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("bad URL: {error}"))
    })?;
    let secure = match uri.scheme_str() {
        Some("https") => true,
        Some("http") => false,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported scheme: {other:?}"),
            ));
        }
    };
    let authority = uri
        .authority()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "URL missing authority"))?;
    if authority.as_str().contains('@') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "HTTP URL user information is not supported",
        ));
    }
    let host = uri
        .host()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "URL missing host"))?
        .to_owned();
    let port = uri.port_u16().unwrap_or(if secure { 443 } else { 80 });
    let path = uri
        .path_and_query()
        .map(|path_and_query| path_and_query.as_str().to_owned())
        .unwrap_or_else(|| "/".to_owned());
    Ok((Origin { secure, host, port }, path))
}

#[derive(Debug)]
struct Reference<'a> {
    scheme: Option<&'a str>,
    authority: Option<&'a str>,
    path: &'a str,
    query: Option<&'a str>,
    fragment: Option<&'a str>,
}

fn resolve_reference(base: &str, reference: &str) -> io::Result<String> {
    let base = http::Uri::from_str(base).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bad base URL: {error}"),
        )
    })?;
    let base_scheme = base
        .scheme_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "base URL missing scheme"))?;
    let base_authority = base.authority().map(http::uri::Authority::as_str);
    let base_path = base.path();
    let base_query = base.query();
    let reference = parse_reference(reference)?;

    let (scheme, authority, path, query) = if let Some(scheme) = reference.scheme {
        (
            scheme,
            reference.authority,
            remove_dot_segments(reference.path),
            reference.query,
        )
    } else if reference.authority.is_some() {
        (
            base_scheme,
            reference.authority,
            remove_dot_segments(reference.path),
            reference.query,
        )
    } else if reference.path.is_empty() {
        (
            base_scheme,
            base_authority,
            base_path.to_owned(),
            reference.query.or(base_query),
        )
    } else {
        let path = if reference.path.starts_with('/') {
            remove_dot_segments(reference.path)
        } else {
            remove_dot_segments(&merge_paths(
                base_authority.is_some(),
                base_path,
                reference.path,
            ))
        };
        (base_scheme, base_authority, path, reference.query)
    };

    Ok(recompose(
        scheme,
        authority,
        &path,
        query,
        reference.fragment,
    ))
}

fn parse_reference(reference: &str) -> io::Result<Reference<'_>> {
    let (without_fragment, fragment) = reference
        .split_once('#')
        .map_or((reference, None), |(target, fragment)| {
            (target, Some(fragment))
        });
    let (scheme, after_scheme) = split_scheme(without_fragment)?;
    let (authority, path_and_query) = split_authority(after_scheme)?;
    let (path, query) = path_and_query
        .split_once('?')
        .map_or((path_and_query, None), |(path, query)| (path, Some(query)));
    Ok(Reference {
        scheme,
        authority,
        path,
        query,
        fragment,
    })
}

fn split_scheme(reference: &str) -> io::Result<(Option<&str>, &str)> {
    let Some(colon) = reference.find(':') else {
        return Ok((None, reference));
    };
    let first_path_delimiter = reference.find(['/', '?']).unwrap_or(reference.len());
    if colon > first_path_delimiter {
        return Ok((None, reference));
    }
    let Some((candidate, rest)) = reference.split_at_checked(colon) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "redirect Location has an invalid scheme boundary",
        ));
    };
    let valid = candidate
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
        && candidate
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'));
    if !valid {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "redirect Location has an invalid scheme",
        ));
    }
    let rest = rest.strip_prefix(':').ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "redirect Location has an invalid scheme separator",
        )
    })?;
    Ok((Some(candidate), rest))
}

fn split_authority(reference: &str) -> io::Result<(Option<&str>, &str)> {
    let Some(remainder) = reference.strip_prefix("//") else {
        return Ok((None, reference));
    };
    let end = remainder.find(['/', '?']).unwrap_or(remainder.len());
    let Some((authority, path_and_query)) = remainder.split_at_checked(end) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "redirect Location has an invalid authority boundary",
        ));
    };
    Ok((Some(authority), path_and_query))
}

fn merge_paths(has_authority: bool, base: &str, reference: &str) -> String {
    if has_authority && base.is_empty() {
        return format!("/{reference}");
    }
    base.rsplit_once('/').map_or_else(
        || reference.to_owned(),
        |(directory, _)| format!("{directory}/{reference}"),
    )
}

fn remove_dot_segments(path: &str) -> String {
    let mut input = path;
    let mut output = String::with_capacity(path.len());
    while !input.is_empty() {
        if let Some(rest) = input
            .strip_prefix("../")
            .or_else(|| input.strip_prefix("./"))
        {
            input = rest;
        } else if input.starts_with("/./") {
            input = input.strip_prefix("/.").unwrap_or(input);
        } else if input == "/." {
            input = "/";
        } else if input.starts_with("/../") {
            input = input.strip_prefix("/..").unwrap_or(input);
            remove_last_segment(&mut output);
        } else if input == "/.." {
            input = "/";
            remove_last_segment(&mut output);
        } else if matches!(input, "." | "..") {
            input = "";
        } else {
            let end = if let Some(rest) = input.strip_prefix('/') {
                rest.find('/').map_or(input.len(), |index| {
                    index.checked_add(1).unwrap_or(input.len())
                })
            } else {
                input.find('/').unwrap_or(input.len())
            };
            let Some((segment, rest)) = input.split_at_checked(end) else {
                break;
            };
            output.push_str(segment);
            input = rest;
        }
    }
    output
}

fn remove_last_segment(path: &mut String) {
    if let Some(slash) = path.rfind('/') {
        path.truncate(slash);
    } else {
        path.clear();
    }
}

fn recompose(
    scheme: &str,
    authority: Option<&str>,
    path: &str,
    query: Option<&str>,
    fragment: Option<&str>,
) -> String {
    let mut target = String::new();
    target.push_str(scheme);
    target.push(':');
    if let Some(authority) = authority {
        target.push_str("//");
        target.push_str(authority);
    }
    target.push_str(path);
    if let Some(query) = query {
        target.push('?');
        target.push_str(query);
    }
    if let Some(fragment) = fragment {
        target.push('#');
        target.push_str(fragment);
    }
    target
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "http://a/b/c/d;p?q";

    #[test]
    fn rfc3986_normal_reference_examples_match_section_5_4_1() {
        let cases = [
            ("g:h", "g:h"),
            ("g", "http://a/b/c/g"),
            ("./g", "http://a/b/c/g"),
            ("g/", "http://a/b/c/g/"),
            ("/g", "http://a/g"),
            ("//g", "http://g"),
            ("?y", "http://a/b/c/d;p?y"),
            ("g?y", "http://a/b/c/g?y"),
            ("#s", "http://a/b/c/d;p?q#s"),
            ("g#s", "http://a/b/c/g#s"),
            ("g?y#s", "http://a/b/c/g?y#s"),
            (";x", "http://a/b/c/;x"),
            ("g;x", "http://a/b/c/g;x"),
            ("g;x?y#s", "http://a/b/c/g;x?y#s"),
            ("", "http://a/b/c/d;p?q"),
            (".", "http://a/b/c/"),
            ("./", "http://a/b/c/"),
            ("..", "http://a/b/"),
            ("../", "http://a/b/"),
            ("../g", "http://a/b/g"),
            ("../..", "http://a/"),
            ("../../", "http://a/"),
            ("../../g", "http://a/g"),
        ];
        for (reference, expected) in cases {
            let resolved = resolve_reference(BASE, reference).expect("RFC reference must resolve");
            assert_eq!(resolved, expected);
        }
    }

    #[test]
    fn rfc3986_abnormal_reference_examples_match_section_5_4_2() {
        let cases = [
            ("../../../g", "http://a/g"),
            ("../../../../g", "http://a/g"),
            ("/./g", "http://a/g"),
            ("/../g", "http://a/g"),
            ("g.", "http://a/b/c/g."),
            (".g", "http://a/b/c/.g"),
            ("g..", "http://a/b/c/g.."),
            ("..g", "http://a/b/c/..g"),
            ("./../g", "http://a/b/g"),
            ("./g/.", "http://a/b/c/g/"),
            ("g/./h", "http://a/b/c/g/h"),
            ("g/../h", "http://a/b/c/h"),
            ("g;x=1/./y", "http://a/b/c/g;x=1/y"),
            ("g;x=1/../y", "http://a/b/c/y"),
            ("g?y/./x", "http://a/b/c/g?y/./x"),
            ("g?y/../x", "http://a/b/c/g?y/../x"),
            ("g#s/./x", "http://a/b/c/g#s/./x"),
            ("g#s/../x", "http://a/b/c/g#s/../x"),
            ("http:g", "http:g"),
        ];
        for (reference, expected) in cases {
            let resolved = resolve_reference(BASE, reference).expect("RFC reference must resolve");
            assert_eq!(resolved, expected);
        }
    }

    #[test]
    fn redirect_policy_preserves_or_rewrites_methods_by_status() {
        assert!(redirects_to_get(301, "POST"));
        assert!(redirects_to_get(302, "post"));
        assert!(redirects_to_get(303, "PUT"));
        assert!(!redirects_to_get(303, "HEAD"));
        assert!(!redirects_to_get(307, "POST"));
        assert!(!redirects_to_get(308, "POST"));
        assert!(!redirects_to_get(302, "PUT"));
    }

    #[test]
    fn redirect_headers_remove_destination_and_connection_specific_fields() {
        let headers = [
            ("Host", "old.test"),
            ("Authorization", "secret"),
            ("Cookie", "session"),
            ("Content-Type", "text/plain"),
            ("Content-Length", "7"),
            ("Connection", "X-Private"),
            ("X-Private", "drop"),
            ("X-Test", "keep"),
        ];
        assert_eq!(
            forwarded_headers(&headers, true, true),
            vec![("X-Test", "keep")]
        );
        assert_eq!(
            forwarded_headers(&headers, false, false),
            vec![
                ("Authorization", "secret"),
                ("Cookie", "session"),
                ("Content-Type", "text/plain"),
                ("Content-Length", "7"),
                ("X-Test", "keep"),
            ]
        );
    }

    #[test]
    fn redirect_location_rejects_empty_unsupported_and_userinfo_targets() {
        let empty = resolve_redirect(BASE, "").expect_err("empty Location must fail");
        assert_eq!(empty.kind(), io::ErrorKind::InvalidData);

        let unsupported = resolve_redirect(BASE, "ftp://example.test/x")
            .expect_err("unsupported redirect scheme must fail");
        assert_eq!(unsupported.kind(), io::ErrorKind::InvalidData);

        let userinfo = resolve_redirect(BASE, "http://user@example.test/x")
            .expect_err("redirect user information must fail");
        assert_eq!(userinfo.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn redirect_fragment_is_not_sent_to_the_origin() {
        let resolved =
            resolve_redirect(BASE, "../g?x=1#section").expect("HTTP redirect must resolve");
        assert_eq!(resolved, "http://a/b/g?x=1");
    }
}
