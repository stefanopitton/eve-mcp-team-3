### What this server does

<!-- One or two sentences: which EO service it wraps and which question it helps answer. -->

Submitted during the **Agentic AI for Earth Observation hackathon** (Berlin, 22 October 2026).

### Tools

<!-- List the tools and what each one returns. -->

### Checklist

- [ ] PR modifies only one server directory under `servers/`
- [ ] `server.py` follows the entrypoint contract (FastMCP settings + streamable-http transport)
- [ ] `requirements.txt` exists and all dependencies have version constraints
- [ ] No `.env` files, credentials, or keys committed (only `.env.template`)
- [ ] No hardcoded secrets in source code (use `os.environ` / `os.getenv`)
- [ ] `.env.template` uses `${PLACEHOLDER}` syntax for secrets (if applicable)
- [ ] No files larger than 5 MB
- [ ] Local test passes (`python test.py` or manual curl test)
- [ ] Docker build succeeds (`docker build -f shared/Dockerfile servers/my-server/`)
- [ ] If server uses credential headers, added header allowlist to `scripts/deploy_server.py`
