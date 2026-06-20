#!/usr/bin/env bash
# =====================================================================
# Push the pump.fun trading agent to GitHub.
#
# Usage:
#   bash scripts/push_to_github.sh <github-url>
#
# Example:
#   bash scripts/push_to_github.sh https://github.com/youruser/pumpfun-agent.git
#
# Or via SSH:
#   bash scripts/push_to_github.sh git@github.com:youruser/pumpfun-agent.git
#
# This script:
#   1. Verifies no secrets are staged
#   2. Adds all project files
#   3. Creates initial commit (or amends if exists)
#   4. Sets the remote
#   5. Pushes to main
# =====================================================================
set -e

REMOTE_URL="${1:-}"
if [[ -z "$REMOTE_URL" ]]; then
  echo "Usage: bash scripts/push_to_github.sh <github-url>"
  echo "Example: bash scripts/push_to_github.sh https://github.com/youruser/pumpfun-agent.git"
  exit 1
fi

cd "$(dirname "$0")/.."

echo "==== 1. Safety check: no secrets staged ===="
# Verify .env and config.yaml are NOT tracked
if git ls-files --error-unmatch .env 2>/dev/null; then
  echo "ERROR: .env is tracked by git. Remove with: git rm --cached .env"
  exit 1
fi
if git ls-files --error-unmatch config/config.yaml 2>/dev/null; then
  echo "ERROR: config/config.yaml is tracked by git. Remove with: git rm --cached config/config.yaml"
  exit 1
fi
echo "  OK: .env and config.yaml are gitignored"

echo
echo "==== 2. Checking gitignore protects sensitive files ===="
for f in .env config/config.yaml data/ *.log *.key *.pem; do
  if git check-ignore -q "$f" 2>/dev/null; then
    echo "  OK: $f is ignored"
  else
    echo "  WARN: $f is not in .gitignore"
  fi
done

echo
echo "==== 3. Stage all project files ===="
git add -A
# Make absolutely sure we don't stage secrets
git reset .env config/config.yaml 2>/dev/null || true
git status --short | head -20

echo
echo "==== 4. Commit ===="
if git log --oneline -1 2>/dev/null; then
  # Repo already has commits; just create a new one
  git commit -m "feat: advanced analyzers (order flow, social graph, MEV, lifecycle, sentiment, alpha signal)

- analysis/order_flow.py: real-time buy/sell pressure, whale detection
- analysis/social_graph.py: wallet clustering, smart money identification
- analysis/mev_detector.py: sandwich attack detection on our txs
- analysis/lifecycle.py: BIRTH/INFANT/ADOLESCENT/MATURE/MIGRATED stages
- analysis/liquidity_depth.py: slippage curve + max position size
- analysis/sentiment.py: Twitter/X + Telegram mentions + hype score
- analysis/alpha_signal.py: weighted ensemble of all 8 analyzers
- Dashboard: /api/analyze/{mint}, /api/order_flow, /api/lifecycle, etc.
- GitHub: CI workflow, CONTRIBUTING, SECURITY, LICENSE" || true
else
  git commit -m "feat: initial commit — pump.fun autonomous trading agent

Multi-strategy (sniping, copy-trade, momentum, grid, anti-rugpull)
Multi-chain (Solana + EVM)
Risk management (fixed + Kelly sizing, daily loss cap, trailing stops)
Smart exits (break-even, TP ladder, dev-sell trigger)
Profit optimization (token scoring, pyramiding, anti-sandwich, Jito MEV)
Advanced analytics (order flow, social graph, lifecycle, sentiment, alpha signal)
Live parameter tuning via dashboard
SQLite persistence
Telegram alerts + dashboard

See README.md for full feature list and setup guide."
fi

echo
echo "==== 5. Set remote ===="
if git remote get-url origin 2>/dev/null; then
  git remote set-url origin "$REMOTE_URL"
  echo "  Updated origin to $REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
  echo "  Added origin: $REMOTE_URL"
fi

echo
echo "==== 6. Push to GitHub ===="
git push -u origin main || git push -u origin master

echo
echo "==== SUCCESS ===="
echo "Your repo is now on GitHub: $REMOTE_URL"
echo
echo "Next steps:"
echo "  1. Add a description on GitHub"
echo "  2. Add topics: pump-fun, solana, trading-bot, memecoin, mev, jito"
echo "  3. Star it if you find it useful"
echo "  4. NEVER commit your .env or config.yaml"
