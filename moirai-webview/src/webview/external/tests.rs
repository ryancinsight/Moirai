use super::*;
use crate::navigation::{Decision, Frame};
use objc2_foundation::NSData;

const SOURCE: &str = "metis://app/index.html";
const DESTINATION: &str = "https://github.com/ryancinsight/metis";

fn capture() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

fn route(links: &ExternalLinks, destination: &str) -> Decision {
    links.decide(
        destination,
        Some(SOURCE),
        true,
        true,
        false,
        Frame::NewWindow,
    )
}

#[test]
fn reads_exact_destinations_from_real_foundation_array() {
    let _trace = capture();
    let _test = tracing::info_span!("native_external_metadata_identity").entered();
    let destinations = [
        DESTINATION,
        "https://webassembly.org/",
        "https://docs.example.org/guide?language=Rust#ownership",
        "https://docs.example.org/guide%20with%20spaces#part%202",
    ];
    let values = tracing::info_span!("arrange").in_scope(|| {
        let values = destinations.map(NSString::from_str);
        NSArray::from_retained_slice(&values)
    });
    let links = tracing::info_span!("act")
        .in_scope(|| metadata_links(Some(&values)).expect("native string array"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(values.len(), destinations.len());
        assert_eq!(
            links,
            ExternalLinks::new(destinations).expect("exact policy oracle")
        );
        for destination in destinations {
            assert_eq!(route(&links, destination), Decision::ExternalPopup);
            assert_eq!(
                route(&links, &format!("{destination}other")),
                Decision::Deny
            );
        }
    });
}

#[test]
fn missing_and_empty_metadata_grant_no_external_links() {
    let _trace = capture();
    let _test = tracing::info_span!("native_external_metadata_empty").entered();
    let empty = tracing::info_span!("arrange").in_scope(NSArray::<NSString>::new);
    let values: [Option<&AnyObject>; 2] = [None, Some(&empty)];
    for value in values {
        let links = tracing::info_span!("act")
            .in_scope(|| metadata_links(value).expect("absent or empty metadata"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(links, ExternalLinks::default());
            assert_eq!(route(&links, DESTINATION), Decision::Deny);
        });
    }
}

#[test]
fn rejects_real_non_array_and_mixed_type_metadata() {
    let _trace = capture();
    let _test = tracing::info_span!("native_external_metadata_types").entered();
    let (text, data, mixed) = tracing::info_span!("arrange").in_scope(|| {
        let text = NSString::from_str(DESTINATION);
        let data = NSData::with_bytes(DESTINATION.as_bytes());
        let mixed = NSArray::<AnyObject>::from_slice(&[&text, &data]);
        (text, data, mixed)
    });
    let scalars: [&AnyObject; 2] = [&text, &data];
    for value in scalars {
        let error = tracing::info_span!("act")
            .in_scope(|| metadata_links(Some(value)).expect_err("array required"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "external links metadata must be an array"
            );
        });
    }
    let error = tracing::info_span!("act")
        .in_scope(|| metadata_links(Some(&mixed)).expect_err("all elements must be strings"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(mixed.len(), 2);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "external links metadata contains a non-string"
        );
    });
}

#[test]
fn native_string_bound_is_inclusive_and_precedes_utf8_copy() {
    let _trace = capture();
    let _test = tracing::info_span!("native_external_metadata_string_bound").entered();
    let (boundary, values) = tracing::info_span!("arrange").in_scope(|| {
        let prefix = "https://docs.example.org/";
        let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let path: String = (0..MAX_EXTERNAL_LINK_BYTES - prefix.len())
            .map(|index| char::from(alphabet[index % alphabet.len()]))
            .collect();
        let boundary = format!("{prefix}{path}");
        let text = NSString::from_str(&boundary);
        let values = NSArray::from_retained_slice(&[text]);
        (boundary, values)
    });
    let links = tracing::info_span!("act")
        .in_scope(|| metadata_links(Some(&values)).expect("inclusive native string bound"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(boundary.len(), MAX_EXTERNAL_LINK_BYTES);
        assert_eq!(values.objectAtIndex(0).length(), MAX_EXTERNAL_LINK_BYTES);
        assert_eq!(route(&links, &boundary), Decision::ExternalPopup);
        assert_eq!(
            route(&links, &boundary[..boundary.len() - 1]),
            Decision::Deny
        );
    });
    for text in [
        format!("{boundary}a"),
        "東".repeat(MAX_EXTERNAL_LINK_BYTES + 1),
    ] {
        let values = tracing::info_span!("arrange")
            .in_scope(|| NSArray::from_retained_slice(&[NSString::from_str(&text)]));
        let error = tracing::info_span!("act")
            .in_scope(|| metadata_links(Some(&values)).expect_err("native copy bound"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                values.objectAtIndex(0).length(),
                MAX_EXTERNAL_LINK_BYTES + 1
            );
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "external links metadata exceeds its string bound"
            );
        });
    }
    let unicode = tracing::info_span!("arrange").in_scope(|| {
        NSArray::from_retained_slice(&[NSString::from_str("https://docs.example.org/東京")])
    });
    let error = tracing::info_span!("act")
        .in_scope(|| metadata_links(Some(&unicode)).expect_err("ASCII byte contract"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "external destination violates the URI byte contract"
        );
    });
}

#[test]
fn native_array_count_bound_is_inclusive() {
    let _trace = capture();
    let _test = tracing::info_span!("native_external_metadata_count_bound").entered();
    let (destinations, values) = tracing::info_span!("arrange").in_scope(|| {
        let destinations: Vec<String> = (0..MAX_EXTERNAL_LINKS)
            .map(|index| format!("https://docs.example.org/chapter/{index}"))
            .collect();
        let values: Vec<_> = destinations
            .iter()
            .map(|value| NSString::from_str(value))
            .collect();
        (destinations, NSArray::from_retained_slice(&values))
    });
    let links = tracing::info_span!("act")
        .in_scope(|| metadata_links(Some(&values)).expect("inclusive native count bound"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(values.len(), MAX_EXTERNAL_LINKS);
        for destination in &destinations {
            assert_eq!(route(&links, destination), Decision::ExternalPopup);
        }
        assert_eq!(
            route(
                &links,
                &format!("https://docs.example.org/chapter/{MAX_EXTERNAL_LINKS}")
            ),
            Decision::Deny
        );
    });
    let overflow = tracing::info_span!("arrange").in_scope(|| {
        let values: Vec<_> = (0..=MAX_EXTERNAL_LINKS)
            .map(|index| NSString::from_str(&format!("https://docs.example.org/chapter/{index}")))
            .collect();
        NSArray::from_retained_slice(&values)
    });
    let error = tracing::info_span!("act")
        .in_scope(|| metadata_links(Some(&overflow)).expect_err("native count overflow"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(overflow.len(), MAX_EXTERNAL_LINKS + 1);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "external links metadata exceeds its count bound"
        );
    });
}

#[test]
fn native_string_array_still_enforces_portable_https_validation() {
    let _trace = capture();
    let _test = tracing::info_span!("native_external_metadata_url_validation").entered();
    for destination in [
        "http://docs.example.org/guide",
        "javascript:alert(1)",
        "https://user:password@docs.example.org/guide",
        "https://docs.example.org:65536/guide",
        "https://docs.example.org/guide%gg",
        "https://docs.example.org/guide#part with spaces",
    ] {
        let values = tracing::info_span!("arrange", destination)
            .in_scope(|| NSArray::from_retained_slice(&[NSString::from_str(destination)]));
        let error = tracing::info_span!("act")
            .in_scope(|| metadata_links(Some(&values)).expect_err("invalid HTTPS metadata"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(values.len(), 1);
            assert_eq!(values.objectAtIndex(0).to_string(), destination);
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        });
    }
}
