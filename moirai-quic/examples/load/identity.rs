//! Shared controlled-probe identity material.

use super::{ProbeResult, invalid};
use moirai_quic::rustls;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};

const CHAIN: &[u8] = include_bytes!("../../../tests/fixtures/localhost-cert.pem");
const KEY: &[u8] = include_bytes!("../../../tests/fixtures/localhost-key.pem");

pub(super) fn certificates() -> ProbeResult<Vec<CertificateDer<'static>>> {
    Ok(CertificateDer::pem_slice_iter(CHAIN).collect::<Result<Vec<_>, _>>()?)
}

pub(super) fn private_key() -> ProbeResult<PrivateKeyDer<'static>> {
    Ok(PrivateKeyDer::from_pem_slice(KEY)?)
}

pub(super) fn roots() -> ProbeResult<rustls::RootCertStore> {
    let certificates = certificates()?;
    let mut roots = rustls::RootCertStore::empty();
    roots.add(
        certificates
            .last()
            .ok_or_else(|| invalid("fixture root is absent"))?
            .clone(),
    )?;
    Ok(roots)
}

pub(super) fn leaf() -> ProbeResult<CertificateDer<'static>> {
    certificates()?
        .into_iter()
        .next()
        .ok_or_else(|| invalid("fixture has no leaf").into())
}
