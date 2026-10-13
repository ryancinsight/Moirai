### Security
- `moirai-http` response parsing now enforces a `max_response_bytes` budget
  (default 64 MiB, configurable via `HttpClient::set_max_response_bytes`). A
  malicious or compromised peer could previously drive unbounded allocation
  (OOM) by advertising a huge `Content-Length`, streaming endless chunked/EOF
  bodies, or trickling header bytes — every read now funnels through one
  size-checked chokepoint, and an oversized `Content-Length` is rejected up front.

