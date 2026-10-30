//! Raw Quinn/Tokio comparison oracle for the authenticated QUIC load probe.
#![cfg(any(unix, windows))]

#[path = "load/reference_entry.rs"]
mod load;

use load::{Phase, Plan, ProbeResult, emit_phase, report};
use std::io::{self, Write};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

fn main() -> ProbeResult<()> {
    let recovery = load::recovery::Recovery::install(false)?;
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=6).contains(&arguments.len()) || !matches!(arguments[0].as_str(), "server" | "client")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "quic_load_tokio server|client ADDR STREAMS MESSAGES FRAME_BYTES [cubic|newreno|bbr-experimental]",
        )
        .into());
    }
    let peer: SocketAddr = arguments[1].parse()?;
    let mut plan = Plan::new(
        arguments[2].parse()?,
        arguments[3].parse()?,
        arguments[4].parse()?,
    )?;
    plan.policy = match arguments.get(5).map(String::as_str).unwrap_or("cubic") {
        "cubic" => moirai_quic::CongestionPolicy::Cubic,
        "newreno" => moirai_quic::CongestionPolicy::NewReno,
        "bbr-experimental" => moirai_quic::CongestionPolicy::ExperimentalBbr,
        _ => {
            return Err(
                io::Error::new(io::ErrorKind::InvalidInput, "unknown congestion policy").into(),
            );
        }
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    runtime.block_on(async {
        if arguments[0] == "server" {
            probe::<true>(peer, plan, &recovery).await
        } else {
            probe::<false>(peer, plan, &recovery).await
        }
    })
}

async fn probe<const RESPONDER: bool>(
    address: SocketAddr,
    plan: Plan,
    recovery: &load::recovery::Recovery,
) -> ProbeResult<()> {
    recovery.emit("start")?;
    let endpoint = load::reference::endpoint::<RESPONDER>(address, plan)?;
    if RESPONDER {
        println!(
            "{}",
            serde_json::json!({"ready": endpoint.local_addr()?.to_string()})
        );
        io::stdout().flush()?;
    }
    let started = Instant::now();
    let outcome = tokio::time::timeout(Duration::from_secs(60), async {
        let exchange = load::reference::run::<RESPONDER>(&endpoint, address, plan, started);
        let observe = recovery.observe(tokio::time::sleep);
        futures::pin_mut!(exchange, observe);
        match futures::future::select(exchange, observe).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right((never, _)) => match never {},
        }
    })
    .await;
    match &outcome {
        Err(error) => {
            tracing::error!(phase = "pair", %error, "load probe fails before endpoint drain")
        }
        Ok(Err(error)) => {
            tracing::error!(phase = "exchange", %error, "load probe fails before endpoint drain")
        }
        Ok(Ok(_)) => {}
    }
    recovery.emit(if matches!(&outcome, Ok(Ok(_))) {
        "success"
    } else {
        "failure"
    })?;
    emit_phase(Phase::Drain);
    endpoint.close(0_u8.into(), b"endpoint closed");
    let drained = tokio::time::timeout(load::reference::OPERATION_TIMEOUT, endpoint.wait_idle())
        .await
        .map_err(|_| load::reference::Error::TimedOut("endpoint drain"));
    if let Err(error) = &drained {
        tracing::error!(phase = "drain", %error, "load probe endpoint drain fails");
    }
    recovery.emit(if drained.is_ok() {
        "drained"
    } else {
        "drain_failure"
    })?;
    let result = outcome.map_err(|_| load::reference::Error::TimedOut("exchange"))??;
    drained?;
    println!(
        "{}",
        serde_json::to_string(&report::<RESPONDER>(plan, result)?)?
    );
    Ok(())
}
