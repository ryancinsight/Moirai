//! Public certificate fixtures are restricted to controlled transport probes.

use super::{Plan, ProbeResult, identity, invalid};
use moirai_quic::{Identity, QuicConfig, QuicConnection, QuicEndpoint};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub(super) fn authorize(connection: &QuicConnection) -> ProbeResult<()> {
    let expected = identity::leaf()?;
    let peer = connection
        .peer_certificates()
        .ok_or_else(|| invalid("mutual TLS peer identity is absent"))?;
    if peer.first() != Some(&expected) || connection.alpn() != b"moirai-load/1" {
        return Err(invalid(
            "authenticated peer differs from the declared probe identity/protocol",
        )
        .into());
    }
    Ok(())
}

pub(crate) fn endpoint<const RESPONDER: bool>(
    address: SocketAddr,
    plan: Plan,
) -> ProbeResult<QuicEndpoint> {
    let identity = Identity::new(identity::certificates()?, identity::private_key()?);
    let roots = identity::roots()?;
    let executor = Arc::new(
        moirai_executor::ExecutorBuilder::new()
            .worker_threads(2)
            .async_threads(1)
            .build()?,
    );
    let config = QuicConfig::new(
        b"moirai-load/1",
        plan.bytes,
        plan.streams,
        1,
        65_536,
        Duration::from_secs(5),
        Duration::from_secs(60),
    )?
    .with_congestion_policy(plan.policy);
    if RESPONDER {
        Ok(QuicEndpoint::server(
            executor,
            address,
            identity,
            Some(roots),
            config,
        )?)
    } else {
        let local = if address.is_ipv6() {
            "[::]:0"
        } else {
            "0.0.0.0:0"
        };
        Ok(QuicEndpoint::client(
            executor,
            local.parse()?,
            roots,
            Some(identity),
            config,
        )?)
    }
}
