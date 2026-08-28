# TLS test fixtures

`localhost-cert.pem` and `localhost-key.pem` are test-only material derived from
Rustls's committed RSA-2048 test CA fixtures at revision
`bcf1c74727b9575208cc019fdc323ea834b19413`:

<https://github.com/rustls/rustls/tree/bcf1c74727b9575208cc019fdc323ea834b19413/test-ca/rsa-2048>

The certificate chain is used only by the in-memory and loopback handshake
integration tests. The private key is intentionally public test material and
must never be reused outside tests or treated as a production credential.

`expired-cert.pem` and `expired-key.pem` are a self-contained chain whose leaf
is **permanently expired**: `notBefore = 2024-01-01`, `notAfter = 2024-12-31`,
signed by a long-lived root (`CN=Moirai Expired-Test Root CA`). They were
generated once with the local OpenSSL 3.6.1 CLI (`openssl ca -startdate
20240101000000Z -enddate 20241231000000Z`); the root key was discarded after
signing and is not committed. The leaf carries `DNS:localhost` and proper
`serverAuth`/`keyUsage` extensions so that hostname validation passes and only
the lifetime check can fail. The socket-level handshake test trusts the valid
root and must reject the chain with rustls's `Expired` certificate error. The
private key is intentionally public test material — never reused outside tests.
