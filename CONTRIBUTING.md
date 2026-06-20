## Contributing

Thanks for your interest in improving this project! Memecoin trading is a
fast-moving adversarial environment, so contributions that improve the bot's
edge, safety, or observability are especially welcome.

### Ways to contribute

- **Bug reports**: Open an issue with reproduction steps, logs, and config
  (redact secrets).
- **New strategies**: Add a strategy in `strategies/` extending `BaseStrategy`.
  Include backtesting data if possible.
- **New analyzers**: Add an analyzer in `analysis/` and wire it into
  `AlphaSignalGenerator`.
- **Chain support**: Add a new chain adapter in `chains/` extending
  `BaseChainAdapter`.
- **Documentation**: Improve README, add examples, fix typos.

### Development setup

```bash
git clone <your-fork>
cd pumpfun-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run in paper mode
cp config/config.yaml.example config/config.yaml
cp .env.example .env
# Edit .env with your secrets
python orchestrator.py
```

### Pull request checklist

- [ ] Code passes `python -m py_compile $(find . -name "*.py")`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] No secrets committed (check with `git diff --cached`)
- [ ] New parameters added to `TUNABLE_SCHEMA` in `utils/config_loader.py`
      so they appear in the dashboard editor
- [ ] README updated if behavior changed
- [ ] No emojis in code unless explicitly requested by maintainer

### Code style

- Python 3.11+
- Type hints everywhere (`from __future__ import annotations`)
- Async-first (use `asyncio` for all I/O)
- Structured logging via `structlog` (see `utils/logger.py`)
- Docstrings on every module and public function
- Max line length: 120 chars

### Commit message convention

```
<type>: <subject>

<body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

Example:
```
feat: add Jito bundle support for MEV-protected sniping

- New chains/jito_client.py with multi-tx bundle builder
- Solana adapter routes large buys through Jito
- Config: jito.enabled, jito.min_size_sol, jito.tip_lamports
```

### Branch naming

- `feat/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`

### Issues

When opening an issue, include:
- Python version
- OS
- Chain (Solana/Base/Ethereum)
- Mode (paper/live)
- Relevant log lines (redact wallet addresses)
- Steps to reproduce

### License

By contributing, you agree your contributions will be licensed under the
MIT license (see `LICENSE`).
