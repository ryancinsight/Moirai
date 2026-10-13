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
    if arguments.len() != 5 || !matches!(arguments[0].as_str(), "server" | "client") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "quic_load server|client ADDR STREAMS MESSAGES FRAME_BYTES",
        )
        .into());
    }
    let peer: SocketAddr = arguments[1].parse()?;
    let plan = Plan::new(
        arguments[2].parse()?,
        arguments[3].parse()?,
        arguments[4].parse()?,
    )?;
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
    endpoint.close();
    moirai_executor::block_on(endpoint.wait_idle())?;
    let result = outcome??;
    println!(
        "{}",
        serde_json::to_string(&report::<RESPONDER>(plan, result)?)?
    );
    Ok(())
}
