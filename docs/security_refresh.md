# /auth/refresh hardening

## Rate limiting

* **Policy**: `/auth/refresh` is protected with a token bucket allowing up to **10 refresh requests per minute**.
* **Scope**: the limiter keys primarily on the bearer token (hashed) and falls back to the caller IP (`X-Forwarded-For` or socket address).
* **Behaviour**:
  * Every attempt is recorded in per-identifier and global counters kept in memory.
  * Once the bucket is empty the endpoint returns `429 Too Many Requests` with a `Retry-After: 60` header.
  * Buckets refill continuously, so the client regains one token every six seconds.

## Anti-replay protections

* Each issued token now includes a cryptographically strong **nonce** recorded alongside the active session.
* When a refresh is requested the service validates that the presented nonce matches the current session nonce and that it has **not been seen before**.
* After a successful refresh the consumed nonce is marked as used and a fresh nonce is embedded into the rotated token.
* Used nonces are retained for at least the token lifetime plus a five minute buffer (900s + 300s by default) to catch delayed replay attempts.
* Nonce history is automatically pruned and removed whenever a session expires or is revoked to avoid unbounded memory usage.

These controls prevent burst refresh attempts from exhausting backend resources and block reuse of captured refresh requests.
