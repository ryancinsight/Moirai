//! Mutually authenticated, byte-verified QUIC load probe; see ADR 0075.
#![cfg(any(unix, windows))]

mod load;

use load::{Plan, ProbeResult, endpoint, report, run};
use std::io::{self, Write};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

fn main() -> ProbeResult<()> {
    tracing_subscriber::fmt().with_writer(io::stderr).init();
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=6).contains(&arguments.len()) || !matches!(arguments[0].as_str(), "server" | "client")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "quic_load server|client ADDR STREAMS MESSAGES FRAME_BYTES [cubic|newreno|bbr-experimental]",
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
    if arguments[0] == "server" {
        probe::<true>(peer, plan)
    } else {
        probe::<false>(peer, plan)
    }
}

fn probe<const RESPONDER: bool>(address: SocketAddr, plan: Plan) -> ProbeResult<()> {
    let endpoint = endpoint::<RESPONDER>(address, plan)?;
    if RESPONDER {
        println!(
            "{}",
            serde_json::json!({"ready": endpoint.local_addr()?.to_string()})
        );
        io::stdout().flush()?;
    }
    let started = Instant::now();
    let outcome = moirai_executor::block_on(moirai_async::timer::timeout(
        Duration::from_secs(60),
        run::<RESPONDER>(&endpoint, address, plan, started),
    ));
    // Preserve the originating error before a peer observes connection close
    // and its controller terminates this process during endpoint drain.
    match &outcome {
        Err(error) => {
            tracing::error!(phase = "pair", %error, "load probe fails before endpoint drain")
        }
        Ok(Err(error)) => {
            tracing::error!(phase = "exchange", %error, "load probe fails before endpoint drain")
        }
        Ok(Ok(_)) => {}
    }
    endpoint.close();
    let drained = moirai_executor::block_on(endpoint.wait_idle());
    if let Err(error) = &drained {
        tracing::error!(phase = "drain", %error, "load probe endpoint drain fails");
    }
    let result = outcome??;
    drained?;
    println!(
        "{}",
        serde_json::to_string(&report::<RESPONDER>(plan, result)?)?
    );
    Ok(())
}
