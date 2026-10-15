use super::Deferred;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn callback_batches_preserve_every_input_and_report_capacity_loss() {
    let _trace = trace();
    let _test = tracing::info_span!("deferred_native_callback").entered();
    let mut queue = tracing::info_span!("arrange").in_scope(Deferred::<_, 3>::new);
    // Real ABI-sized values distinguish event kind, physical scan, release,
    // Unicode-surrogate and HRESULT dimensions; no constant result can pass.
    let records = [(0x100_i32, 0xe01d_u32), (0x101, 0xe01d), (0x102, 0xd83e)];
    let (actual, omitted) = tracing::info_span!("act").in_scope(|| {
        for record in records {
            queue.push(record);
        }
        queue.push((0x102, 0xdd80));
        queue.push((-2_147_467_259, 0));
        queue.take()
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(actual, records.map(Some));
        assert_eq!(omitted, 2);
        assert_eq!(queue.take(), ([None; 3], 0));
    });
    let next = (0x101, 0x1d);
    tracing::info_span!("act").in_scope(|| queue.push(next));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(queue.take(), ([Some(next), None, None], 0));
    });
}

#[test]
fn empty_and_single_record_capacities_have_exact_boundaries() {
    let _trace = trace();
    let _test = tracing::info_span!("deferred_callback_boundaries").entered();
    let (mut empty, mut single) = tracing::info_span!("arrange")
        .in_scope(|| (Deferred::<u16, 0>::new(), Deferred::<u16, 1>::new()));
    tracing::info_span!("act").in_scope(|| {
        empty.push(0xd83e);
        single.push(0xdd80);
        single.push(0x41);
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(empty.take(), ([], 1));
        assert_eq!(single.take(), ([Some(0xdd80)], 1));
    });
}
