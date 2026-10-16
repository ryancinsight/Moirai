//! A prefix of six insertions owes exactly the unmatched inverse operations.

pub(super) fn releases(sent: u32) -> impl Iterator<Item = usize> {
    [(2, 3), (1, 4), (0, 5)]
        .into_iter()
        .filter(move |(down, up)| *down < sent && sent <= *up)
        .map(|(_, up)| usize::try_from(up).expect("six-stroke inverse index"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_admitted_prefix_releases_only_owed_keys_in_reverse_press_order() {
        let _trace = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_test_writer()
                .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
                .finish(),
        );
        let expected = tracing::info_span!("arrange").in_scope(|| {
            [
                vec![],
                vec![5],
                vec![4, 5],
                vec![3, 4, 5],
                vec![4, 5],
                vec![5],
                vec![],
            ]
        });
        for (sent, expected) in expected.into_iter().enumerate() {
            let actual: Vec<_> = tracing::info_span!("act", sent).in_scope(|| {
                super::releases(u32::try_from(sent).expect("six-stroke prefix")).collect()
            });
            tracing::info_span!("assert", sent).in_scope(|| assert_eq!(actual, expected));
        }
    }
}
