use super::*;

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

#[test]
fn admits_only_exact_host_configured_destinations() {
    let _trace = capture();
    let _test = tracing::info_span!("external_destination_identity").entered();
    let destinations = [
        DESTINATION,
        "https://webassembly.org/",
        "https://docs.example.org:443/guide?language=Rust#ownership",
        "https://docs.example.org/guide%20with%20spaces#part%202",
        "https://[::1]:65535/guide",
    ];
    let links = tracing::info_span!("arrange")
        .in_scope(|| ExternalLinks::new(destinations).expect("valid exact destinations"));
    for destination in destinations {
        let decision = tracing::info_span!("act", destination).in_scope(|| {
            links.decide(
                destination,
                Some(SOURCE),
                true,
                true,
                false,
                Frame::NewWindow,
            )
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(decision, Decision::ExternalPopup);
            assert!(
                links
                    .destinations
                    .iter()
                    .any(|value| &**value == destination)
            );
        });
    }
    for destination in [
        "https://github.com/",
        "https://github.com/ryancinsight/metis/",
        "https://github.com/ryancinsight/metis?tab=readme",
        "https://github.com/ryancinsight/metis#readme",
        "https://github.com/ryancinsight/METIS",
        "https://GITHUB.com/ryancinsight/metis",
        "https://github.com:443/ryancinsight/metis",
        "https://github.com.evil.example/ryancinsight/metis",
        "https://webassembly.org",
        "https://docs.example.org:443/guide?language=Rust#borrowing",
        "https://docs.example.org/guide%20with%20spaces#part%203",
    ] {
        let decision = tracing::info_span!("act", destination).in_scope(|| {
            links.decide(
                destination,
                Some(SOURCE),
                true,
                true,
                false,
                Frame::NewWindow,
            )
        });
        tracing::info_span!("assert").in_scope(|| assert_eq!(decision, Decision::Deny));
    }
}

#[test]
fn rejects_non_https_malformed_and_unsafe_destinations() {
    let _trace = capture();
    let _test = tracing::info_span!("external_destination_validation").entered();
    for destination in [
        "",
        "/guide",
        "//docs.example.org/guide",
        "https:docs.example.org/guide",
        "http://docs.example.org/guide",
        "javascript:alert(1)",
        "data:text/html,external",
        "file:///private/guide",
        "mailto:ryan@example.org",
        "metis://app/index.html",
        "https://user:password@docs.example.org/guide",
        "https://user@docs.example.org/guide",
        "https://docs.example.org@evil.example/guide",
        "https://:443/guide",
        "https:///guide",
        "https://docs.example.org:/guide",
        "https://docs.example.org:abc/guide",
        "https://docs.example.org:+443/guide",
        "https://docs.example.org:65536/guide",
        "https://docs.example.org:443:443/guide",
        "https://docs.example.org/guide%",
        "https://docs.example.org/guide%2",
        "https://docs.example.org/guide%gg",
        "https://docs.example.org/guide#part%g0",
        "https://docs.example.org/guide#part#other",
        "https://docs.example.org\\evil/guide",
        "https://docs.example.org/guide with spaces",
        "https://docs.example.org/guide\twith-tabs",
        "https://docs.example.org/guide\nnext-line",
        "https://docs.example.org/guide\rnext-line",
        "https://docs.example.org/guide\0suffix",
        "https://docs.example.org/guide\x7f",
        "https://docs.example.org/東京",
        "https://docs.example.org/guide#part with spaces",
        "https://docs.example.org/<guide>",
        "https://docs.example.org/\"guide\"",
    ] {
        tracing::info_span!("arrange", destination).in_scope(|| {
            assert!(destination.len() <= MAX_EXTERNAL_LINK_BYTES);
        });
        let error = tracing::info_span!("act")
            .in_scope(|| ExternalLinks::new([destination]).expect_err("rejected destination"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        });
    }
}

#[test]
fn enforces_source_action_download_and_target_contract() {
    let _trace = capture();
    let _test = tracing::info_span!("external_navigation_partition").entered();
    let links = tracing::info_span!("arrange")
        .in_scope(|| ExternalLinks::new([DESTINATION]).expect("exact destination"));
    for (source, packaged_source) in [
        (Some(SOURCE), true),
        (Some("metis://app/other.html?state=ready#link"), true),
        (None, false),
        (Some("https://github.com/ryancinsight/metis"), false),
        (Some("metis://app.evil/index.html"), false),
        (Some("metis://app@evil/index.html"), false),
        (Some("metis://other/index.html"), false),
    ] {
        for main_source in [false, true] {
            for link_activated in [false, true] {
                for download in [false, true] {
                    for target in [Frame::Main, Frame::Subframe, Frame::NewWindow] {
                        let expected = if packaged_source
                            && main_source
                            && link_activated
                            && !download
                            && target == Frame::NewWindow
                        {
                            Decision::ExternalPopup
                        } else {
                            Decision::Deny
                        };
                        let decision = tracing::info_span!(
                            "act",
                            ?source,
                            main_source,
                            link_activated,
                            download,
                            ?target
                        )
                        .in_scope(|| {
                            links.decide(
                                DESTINATION,
                                source,
                                main_source,
                                link_activated,
                                download,
                                target,
                            )
                        });
                        tracing::info_span!("assert").in_scope(|| assert_eq!(decision, expected));
                    }
                }
            }
        }
    }
}

#[test]
fn packaged_existing_frames_remain_admitted_and_empty_policy_denies_external() {
    let _trace = capture();
    let _test = tracing::info_span!("packaged_navigation_preservation").entered();
    for links in tracing::info_span!("arrange").in_scope(|| {
        [
            ExternalLinks::default(),
            ExternalLinks::new([] as [&str; 0]).expect("empty policy"),
        ]
    }) {
        for destination in [SOURCE, "metis://app/other.html?revision=2#part"] {
            for target in [Frame::Main, Frame::Subframe, Frame::NewWindow] {
                let decision = tracing::info_span!("act", destination, ?target)
                    .in_scope(|| links.decide(destination, None, false, false, false, target));
                let expected = if target == Frame::NewWindow {
                    Decision::Deny
                } else {
                    Decision::Allow
                };
                tracing::info_span!("assert").in_scope(|| assert_eq!(decision, expected));
            }
        }
        let decision = tracing::info_span!("act").in_scope(|| {
            links.decide(
                DESTINATION,
                Some(SOURCE),
                true,
                true,
                false,
                Frame::NewWindow,
            )
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(decision, Decision::Deny);
            assert_eq!(links.destinations.len(), 0);
        });
    }
}

#[test]
fn byte_and_entry_bounds_are_inclusive_without_truncating_values() {
    let _trace = capture();
    let _test = tracing::info_span!("external_policy_bounds").entered();
    let (boundary, destinations) = tracing::info_span!("arrange").in_scope(|| {
        let prefix = "https://docs.example.org/";
        let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let path: String = (0..MAX_EXTERNAL_LINK_BYTES - prefix.len())
            .map(|index| char::from(alphabet[index % alphabet.len()]))
            .collect();
        let boundary = format!("{prefix}{path}");
        let destinations: Vec<String> = (0..MAX_EXTERNAL_LINKS)
            .map(|index| format!("https://docs.example.org/chapter/{index}"))
            .collect();
        (boundary, destinations)
    });
    let (bounded, counted) = tracing::info_span!("act").in_scope(|| {
        (
            ExternalLinks::new([&boundary]).expect("inclusive byte bound"),
            ExternalLinks::new(&destinations).expect("inclusive entry bound"),
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(boundary.len(), MAX_EXTERNAL_LINK_BYTES);
        assert_eq!(&*bounded.destinations[0], boundary);
        assert_eq!(counted.destinations.len(), MAX_EXTERNAL_LINKS);
        for destination in &destinations {
            assert_eq!(
                counted.decide(
                    destination,
                    Some(SOURCE),
                    true,
                    true,
                    false,
                    Frame::NewWindow
                ),
                Decision::ExternalPopup
            );
        }
    });
    for destinations in [
        vec![format!("{boundary}a")],
        (0..=MAX_EXTERNAL_LINKS)
            .map(|index| format!("https://docs.example.org/chapter/{index}"))
            .collect(),
        vec![DESTINATION.to_owned(); MAX_EXTERNAL_LINKS + 1],
    ] {
        let error = tracing::info_span!("act")
            .in_scope(|| ExternalLinks::new(destinations).expect_err("exclusive upper violation"));
        tracing::info_span!("assert")
            .in_scope(|| assert_eq!(error.kind(), io::ErrorKind::InvalidInput));
    }
}

#[test]
fn retains_distinct_spelling_and_deduplicates_exact_entries() {
    let _trace = capture();
    let _test = tracing::info_span!("external_policy_retained_values").entered();
    let destinations = tracing::info_span!("arrange").in_scope(|| {
        [
            DESTINATION,
            DESTINATION,
            "https://github.com/ryancinsight/metis#readme",
            "https://github.com/ryancinsight/metis%2Fguide",
        ]
    });
    let links = tracing::info_span!("act")
        .in_scope(|| ExternalLinks::new(destinations).expect("validated destinations"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            links
                .destinations
                .iter()
                .map(|value| &**value)
                .collect::<Vec<_>>(),
            [destinations[0], destinations[2], destinations[3]]
        );
    });
}
