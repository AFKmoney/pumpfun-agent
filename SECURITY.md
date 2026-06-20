## Security Policy

### Reporting a Vulnerability

If you discover a security vulnerability, please DO NOT open a public issue.
Instead, email: security@example.com (replace with your real email).

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Suggested fix (if any)

We will respond within 48 hours.

### Known Security Considerations

This software handles **real money** on Solana and EVM chains. Read the
following carefully:

1. **Private keys**: NEVER commit your `.env` file or `config.yaml`. The
   `.gitignore` excludes both. Double-check before pushing.

2. **Wallet isolation**: Use a dedicated wallet with only the funds you can
   afford to lose. Do NOT reuse your main wallet.

3. **API keys**: All API keys (Helius, Birdeye, Twitter, Telegram) are
   loaded from environment variables. Rotate them if leaked.

4. **Slippage**: Default slippage is 8% (800 bps). This means a $100 buy
   could execute at up to $108. The AntiSandwich module reduces this for
   small trades. Tune via the dashboard.

5. **Kill switch**: Three independent kill switches exist — file sentinel,
   Telegram `/kill`, dashboard button. Use any of them if anything looks
   suspicious.

6. **Audit log**: All config changes made via the dashboard are logged to
   SQLite. Review the audit log if you notice unexpected behavior.
