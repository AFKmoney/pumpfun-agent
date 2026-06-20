"""
dashboard.py
=============
Lightweight FastAPI dashboard to monitor the agent in real time + tune
all parameters without restarting.

Endpoints:
- GET /                  -> HTML dashboard (auto-refresh 5s, parameter editor included)
- GET /api/status        -> agent status (state, daily PnL, kill switch)
- GET /api/positions     -> open positions
- GET /api/trades        -> last 100 trades
- GET /api/blacklist     -> current blacklist
- GET /api/config        -> all tunable parameters with current values + schema
- POST /api/config       -> batch update tunable parameters (hot-reload)
- POST /api/config/reset -> reset a single parameter to its current YAML value
- GET /api/config/audit  -> audit log of recent config changes
- POST /api/kill         -> trigger kill switch manually (protected by token)

Run standalone:
    python -m utils.dashboard
Or auto-started by orchestrator when monitoring.enabled = true.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from utils.config_loader import Config, TUNABLE_SCHEMA
from utils.config_hot_reload import ConfigHotReloader
from utils.kill_switch import KillSwitch
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("dashboard")


# =====================================================================
# HTML (single-page app with built-in styling + JS)
# =====================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pump.fun Agent Dashboard</title>
<style>
  body { font-family: 'Inter', system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
  .header { background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 10; }
  .header h1 { margin: 0; font-size: 20px; }
  .nav { display: flex; gap: 8px; }
  .nav button { background: #334155; color: #cbd5e1; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .nav button.active { background: #38bdf8; color: #0f172a; font-weight: 600; }
  .nav button:hover { background: #475569; }
  .nav button.active:hover { background: #0ea5e9; }
  .status-pill { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-left: 12px; }
  .pill-running { background: #16a34a; color: white; }
  .pill-stopped { background: #dc2626; color: white; }
  .pill-killed  { background: #f59e0b; color: black; }
  .btn-danger { background: #dc2626; color: white; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .btn-danger:hover { background: #b91c1c; }
  .btn-primary { background: #38bdf8; color: #0f172a; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600; }
  .btn-primary:hover { background: #0ea5e9; }
  .view { display: none; padding: 24px; max-width: 1400px; margin: 0 auto; }
  .view.active { display: block; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155; }
  .card h3 { margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase; color: #94a3b8; }
  .card .value { font-size: 24px; font-weight: 700; }
  .positive { color: #4ade80; }
  .negative { color: #f87171; }
  table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 16px; }
  th { background: #334155; text-align: left; padding: 10px 12px; font-size: 12px; text-transform: uppercase; }
  td { padding: 10px 12px; border-bottom: 1px solid #334155; font-size: 13px; }
  .token { font-family: 'JetBrains Mono', monospace; color: #38bdf8; }
  .small { font-size: 11px; color: #94a3b8; }
  .section-title { font-size: 14px; font-weight: 600; margin: 24px 0 12px 0; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
  .config-group { background: #1e293b; padding: 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #334155; }
  .config-group h4 { margin: 0 0 12px 0; font-size: 14px; color: #38bdf8; padding-bottom: 8px; border-bottom: 1px solid #334155; }
  .config-row { display: grid; grid-template-columns: 1fr 180px 24px; gap: 12px; padding: 8px 0; border-bottom: 1px solid #2d3a4f; align-items: center; }
  .config-row:last-child { border-bottom: none; }
  .config-label { font-size: 13px; }
  .config-label .desc { font-size: 11px; color: #94a3b8; display: block; margin-top: 2px; }
  .config-input { background: #0f172a; border: 1px solid #475569; color: #e2e8f0; padding: 6px 10px; border-radius: 4px; font-size: 13px; font-family: 'JetBrains Mono', monospace; width: 100%; }
  .config-input:focus { outline: none; border-color: #38bdf8; }
  .config-input.dirty { border-color: #f59e0b; }
  .config-range { font-size: 10px; color: #64748b; margin-top: 2px; }
  .reset-btn { background: transparent; color: #64748b; border: none; cursor: pointer; font-size: 14px; padding: 0; }
  .reset-btn:hover { color: #f87171; }
  .toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 20px; border-radius: 6px; font-size: 13px; z-index: 100; opacity: 0; transition: opacity 0.3s; }
  .toast.show { opacity: 1; }
  .toast.success { background: #16a34a; color: white; }
  .toast.error { background: #dc2626; color: white; }
  .toggle { position: relative; display: inline-block; width: 40px; height: 22px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: #475569; border-radius: 22px; transition: 0.3s; }
  .toggle .slider:before { position: absolute; content: ''; height: 16px; width: 16px; left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: 0.3s; }
  .toggle input:checked + .slider { background: #16a34a; }
  .toggle input:checked + .slider:before { transform: translateX(18px); }
  .save-bar { position: sticky; bottom: 0; background: #1e293b; padding: 12px 24px; border-top: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; margin: 24px -24px -24px -24px; }
  .dirty-count { font-size: 13px; color: #f59e0b; }
  .audit-entry { padding: 6px 0; border-bottom: 1px solid #2d3a4f; font-size: 12px; }
  .audit-path { color: #38bdf8; font-family: monospace; }
  .audit-arrow { color: #94a3b8; }
  .audit-time { color: #64748b; font-size: 11px; }
</style>
</head>
<body>
<div class="header">
  <h1>🎯 Pump.fun Trading Agent</h1>
  <div style="display: flex; align-items: center;">
    <nav class="nav">
      <button class="nav-btn active" data-view="monitoring">📊 Monitoring</button>
      <button class="nav-btn" data-view="params">⚙️ Parameters</button>
      <button class="nav-btn" data-view="audit">📜 Audit Log</button>
    </nav>
    <span id="status-pill" class="status-pill">...</span>
    <button class="btn-danger" style="margin-left: 12px;" onclick="if(confirm('Kill agent?')) fetch('/api/kill', {method:'POST'}).then(r=>r.json()).then(()=>location.reload())">KILL</button>
  </div>
</div>

<!-- ============ MONITORING VIEW ============ -->
<div id="view-monitoring" class="view active">
  <div class="grid">
    <div class="card"><h3>State</h3><div class="value" id="state">...</div></div>
    <div class="card"><h3>Open Positions</h3><div class="value" id="positions">...</div></div>
    <div class="card"><h3>Daily PnL</h3><div class="value" id="pnl">...</div></div>
    <div class="card"><h3>Kill Switch</h3><div class="value" id="kill">...</div></div>
    <div class="card"><h3>Blacklisted Tokens</h3><div class="value" id="blacklist-count">...</div></div>
    <div class="card"><h3>Last Trade</h3><div class="value small" id="last-trade">...</div></div>
  </div>

  <div class="section-title">Open Positions</div>
  <table id="positions-table">
    <thead><tr><th>Chain</th><th>Token</th><th>Strategy</th><th>Entry</th><th>Size (SOL)</th><th>Opened</th><th>SL/TP</th></tr></thead>
    <tbody></tbody>
  </table>

  <div class="section-title">Recent Trades</div>
  <table id="trades-table">
    <thead><tr><th>Time</th><th>Chain</th><th>Token</th><th>Strategy</th><th>Side</th><th>Size</th><th>Price</th><th>PnL %</th></tr></thead>
    <tbody></tbody>
  </table>

  <div class="section-title">Blacklist</div>
  <table id="blacklist-table">
    <thead><tr><th>Token</th><th>Expires</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<!-- ============ PARAMETERS VIEW ============ -->
<div id="view-params" class="view">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
    <div>
      <h2 style="margin: 0 0 4px 0;">⚙️ Parameter Tuning</h2>
      <p class="small" style="margin: 0;">Changes apply instantly via hot-reload. No restart needed. Backups saved to config.yaml.bak</p>
    </div>
  </div>
  <div id="params-container">Loading parameters...</div>
  <div class="save-bar">
    <span class="dirty-count" id="dirty-count">0 changes</span>
    <button class="btn-primary" id="save-btn" onclick="saveChanges()" disabled>Save & Apply</button>
  </div>
</div>

<!-- ============ AUDIT LOG VIEW ============ -->
<div id="view-audit" class="view">
  <h2 style="margin: 0 0 16px 0;">📜 Config Change Audit Log</h2>
  <div id="audit-container">Loading...</div>
</div>

<div id="toast" class="toast"></div>

<script>
const dirtyParams = new Set();
let allParams = {};

// ---- Navigation ----
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('view-' + btn.dataset.view).classList.add('active');
    if (btn.dataset.view === 'params') loadParams();
    if (btn.dataset.view === 'audit') loadAudit();
  });
});

// ---- Monitoring refresh ----
function fmt(s, n) { return (s||'').slice(0, n||8); }
async function refreshMonitoring() {
  try {
    const [status, positions, trades, blacklist] = await Promise.all([
      fetch('/api/status').then(r=>r.json()),
      fetch('/api/positions').then(r=>r.json()),
      fetch('/api/trades').then(r=>r.json()),
      fetch('/api/blacklist').then(r=>r.json()),
    ]);
    document.getElementById('state').textContent = status.state;
    document.getElementById('positions').textContent = status.open_positions;
    const pnlEl = document.getElementById('pnl');
    pnlEl.textContent = (status.daily_pnl_pct||0).toFixed(2) + '%';
    pnlEl.className = 'value ' + ((status.daily_pnl_pct||0) >= 0 ? 'positive' : 'negative');
    const killEl = document.getElementById('kill');
    killEl.textContent = status.kill_switch ? 'ACTIVE' : 'inactive';
    killEl.className = 'value ' + (status.kill_switch ? 'negative' : 'positive');
    const pill = document.getElementById('status-pill');
    pill.textContent = status.state;
    pill.className = 'status-pill ' + (status.state==='RUNNING'?'pill-running':status.kill_switch?'pill-killed':'pill-stopped');
    document.getElementById('blacklist-count').textContent = blacklist.length;
    document.getElementById('last-trade').textContent = trades.length ? new Date(trades[0].ts*1000).toLocaleTimeString() : '-';

    const ptb = document.querySelector('#positions-table tbody');
    ptb.innerHTML = positions.map(p => `<tr>
      <td>${p.chain}</td><td class="token">${fmt(p.token, 8)}</td>
      <td>${p.strategy}</td><td>${p.entry_price.toFixed(8)}</td>
      <td>${p.size_base.toFixed(4)}</td>
      <td class="small">${new Date(p.opened_at*1000).toLocaleString()}</td>
      <td class="small">${p.stop_loss_pct}% / ${p.take_profit_pct}%</td>
    </tr>`).join('') || '<tr><td colspan=7 class="small">No open positions</td></tr>';

    const ttb = document.querySelector('#trades-table tbody');
    ttb.innerHTML = trades.slice(0, 50).map(t => `<tr>
      <td class="small">${new Date(t.ts*1000).toLocaleString()}</td>
      <td>${t.chain}</td><td class="token">${fmt(t.token, 8)}</td>
      <td>${t.strategy}</td><td>${t.side}</td>
      <td>${t.size_base.toFixed(4)}</td><td>${t.price.toFixed(8)}</td>
      <td class="${(t.pnl_pct||0)>=0?'positive':'negative'}">${t.pnl_pct!==null?t.pnl_pct.toFixed(2)+'%':'-'}</td>
    </tr>`).join('') || '<tr><td colspan=8 class="small">No trades yet</td></tr>';

    const bltb = document.querySelector('#blacklist-table tbody');
    bltb.innerHTML = blacklist.map(b => `<tr>
      <td class="token">${fmt(b.token, 12)}</td>
      <td class="small">${new Date(b.expiry_ts*1000).toLocaleString()}</td>
    </tr>`).join('') || '<tr><td colspan=2 class="small">Empty</td></tr>';
  } catch (e) { console.error(e); }
}

// ---- Parameters editor ----
async function loadParams() {
  try {
    const resp = await fetch('/api/config').then(r=>r.json());
    allParams = resp.params;
    dirtyParams.clear();
    renderParams();
    updateDirtyCount();
  } catch (e) {
    document.getElementById('params-container').innerHTML = '<p class="small">Failed to load: ' + e + '</p>';
  }
}

function renderParams() {
  // Group params by top-level section
  const groups = {};
  for (const [path, info] of Object.entries(allParams)) {
    const section = path.split('.')[0];
    if (!groups[section]) groups[section] = [];
    groups[section].push({path, ...info});
  }
  const groupLabels = {
    trading: '💱 Trading',
    risk: '🛡️ Risk Management',
    smart_exits: '🎯 Smart Exits',
    pyramiding: '📈 Pyramiding',
    scoring: '⭐ Token Scoring',
    anti_sandwich: '🥪 Anti-Sandwich',
    jito: '⚡ Jito (MEV Protection)',
    strategies: '🤖 Strategies',
  };
  let html = '';
  for (const [section, items] of Object.entries(groups)) {
    html += `<div class="config-group"><h4>${groupLabels[section]||section} (${items.length})</h4>`;
    for (const item of items) {
      const isBool = item.type === 'bool';
      const value = item.current;
      const rangeText = (item.min !== null && item.max !== null) ? `min: ${item.min}, max: ${item.max}` : '';
      if (isBool) {
        html += `<div class="config-row">
          <div class="config-label">${item.path.split('.').slice(1).join('.')}<span class="desc">${item.desc}</span></div>
          <label class="toggle"><input type="checkbox" data-path="${item.path}" ${value?'checked':''} onchange="markDirty('${item.path}', this.checked)"><span class="slider"></span></label>
          <button class="reset-btn" onclick="resetParam('${item.path}')" title="Reset">↺</button>
        </div>`;
      } else {
        html += `<div class="config-row">
          <div class="config-label">${item.path.split('.').slice(1).join('.')}<span class="desc">${item.desc}</span><span class="config-range">${rangeText}</span></div>
          <input class="config-input" type="${item.type==='int'?'number':'number'}" step="${item.type==='int'?'1':'0.01'}" data-path="${item.path}" value="${value ?? ''}" oninput="markDirty('${item.path}', this.value)" ${item.min!==null?'min="'+item.min+'"':''} ${item.max!==null?'max="'+item.max+'"':''}>
          <button class="reset-btn" onclick="resetParam('${item.path}')" title="Reset">↺</button>
        </div>`;
      }
    }
    html += '</div>';
  }
  document.getElementById('params-container').innerHTML = html;
}

function markDirty(path, value) {
  const input = document.querySelector(`[data-path="${path}"]`);
  const original = allParams[path]?.current;
  const currentVal = input.type === 'checkbox' ? input.checked : (input.type === 'number' ? parseFloat(input.value) : input.value);
  if (currentVal === original || String(currentVal) === String(original)) {
    dirtyParams.delete(path);
    if (input.classList) input.classList.remove('dirty');
  } else {
    dirtyParams.add(path);
    if (input.classList) input.classList.add('dirty');
  }
  updateDirtyCount();
}

function resetParam(path) {
  const input = document.querySelector(`[data-path="${path}"]`);
  const original = allParams[path]?.current;
  if (input.type === 'checkbox') input.checked = original;
  else input.value = original;
  dirtyParams.delete(path);
  if (input.classList) input.classList.remove('dirty');
  updateDirtyCount();
}

function updateDirtyCount() {
  const n = dirtyParams.size;
  document.getElementById('dirty-count').textContent = n + (n === 1 ? ' change' : ' changes');
  document.getElementById('save-btn').disabled = (n === 0);
}

async function saveChanges() {
  if (dirtyParams.size === 0) return;
  const updates = {};
  for (const path of dirtyParams) {
    const input = document.querySelector(`[data-path="${path}"]`);
    updates[path] = input.type === 'checkbox' ? input.checked : parseFloat(input.value);
  }
  if (!confirm(`Apply ${Object.keys(updates).length} parameter changes? This will hot-reload live modules.`)) return;
  try {
    const resp = await fetch('/api/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({updates}),
    }).then(r=>r.json());
    let okCount = 0, failCount = 0;
    let failMsgs = [];
    for (const r of resp.results) {
      if (r.success) okCount++;
      else { failCount++; failMsgs.push(`${r.path}: ${r.message}`); }
    }
    if (failCount === 0) {
      showToast(`✓ ${okCount} parameters updated and applied live`, 'success');
      dirtyParams.clear();
      await loadParams();
    } else {
      showToast(`⚠ ${okCount} ok, ${failCount} failed: ${failMsgs.join('; ')}`, 'error');
    }
  } catch (e) {
    showToast('Save failed: ' + e, 'error');
  }
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  setTimeout(() => t.className = 'toast ' + type, 4000);
}

// ---- Audit log ----
async function loadAudit() {
  try {
    const entries = await fetch('/api/config/audit').then(r=>r.json());
    const container = document.getElementById('audit-container');
    if (!entries.length) {
      container.innerHTML = '<p class="small">No config changes logged yet.</p>';
      return;
    }
    container.innerHTML = entries.map(e => `<div class="audit-entry">
      <span class="audit-time">${new Date(e.ts*1000).toLocaleString()}</span>
      &nbsp;<span class="audit-path">${e.path}</span>
      &nbsp;<span class="audit-arrow">${e.old_value} →</span>
      &nbsp;<strong>${e.new_value}</strong>
      &nbsp;<span class="small">[${e.source}]</span>
    </div>`).join('');
  } catch (e) {
    document.getElementById('audit-container').innerHTML = '<p class="small">Failed: ' + e + '</p>';
  }
}

// ---- Init ----
refreshMonitoring();
setInterval(refreshMonitoring, 5000);
</script>
</body>
</html>
"""


# =====================================================================
# Pydantic models
# =====================================================================
class ConfigUpdateRequest(BaseModel):
    updates: dict[str, object]


# =====================================================================
# App factory
# =====================================================================
def build_app() -> FastAPI:
    cfg = Config.get()
    dash_token = os.environ.get("DASHBOARD_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("dashboard.starting")
        yield
        log.info("dashboard.stopping")

    app = FastAPI(title="Pump.fun Agent Dashboard", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _orchestrator():
        try:
            from orchestrator import Orchestrator
            return Orchestrator._instance
        except Exception:
            return None

    def _get_hot_reloader():
        orch = _orchestrator()
        if orch and hasattr(orch, 'hot_reloader'):
            return orch.hot_reloader
        # Standalone mode (no orchestrator) — create a reloader without orch
        return ConfigHotReloader(orchestrator=None)

    def _check_token(token: Optional[str]) -> None:
        if dash_token and token != dash_token:
            raise HTTPException(status_code=401, detail="Invalid token")

    def _format_position(p) -> dict:
        return {
            "chain": p.get("chain") if isinstance(p, dict) else p.chain,
            "token": p.get("token") if isinstance(p, dict) else p.token,
            "strategy": p.get("strategy") if isinstance(p, dict) else p.strategy,
            "entry_price": p.get("entry_price") if isinstance(p, dict) else p.entry_price,
            "size_base": p.get("size_base") if isinstance(p, dict) else p.size_base,
            "size_token": p.get("size_token") if isinstance(p, dict) else p.size_token,
            "opened_at": p.get("opened_at") if isinstance(p, dict) else p.opened_at,
            "stop_loss_pct": p.get("stop_loss_pct") if isinstance(p, dict) else p.stop_loss_pct,
            "take_profit_pct": p.get("take_profit_pct") if isinstance(p, dict) else p.take_profit_pct,
        }

    # ------------------------------------------------------------------
    # Monitoring routes
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return DASHBOARD_HTML

    @app.get("/api/status")
    async def api_status():
        orch = _orchestrator()
        if orch:
            return {
                "state": orch.state,
                "open_positions": len(orch.risk.positions),
                "daily_pnl_pct": orch.risk.daily_pnl_pct,
                "kill_switch": KillSwitch.is_triggered(),
                "kill_reason": KillSwitch.reason(),
                "strategies": [s.name for s in orch.strategies],
            }
        return {
            "state": "NOT_STARTED",
            "open_positions": 0,
            "daily_pnl_pct": 0,
            "kill_switch": KillSwitch.is_triggered(),
        }

    @app.get("/api/positions")
    async def api_positions():
        db = Persistence.get()
        return [_format_position(p) for p in db.load_open_positions().values()]

    @app.get("/api/trades")
    async def api_trades(limit: int = Query(100, le=500)):
        db = Persistence.get()
        return db.load_recent_trades(limit=limit)

    @app.get("/api/blacklist")
    async def api_blacklist():
        db = Persistence.get()
        return db.load_blacklist()

    @app.get("/api/wallets")
    async def api_wallets():
        from utils.wallet_manager import WalletManager
        wm = WalletManager()
        return wm.redacted_summary()

    # ------------------------------------------------------------------
    # Config tuning routes
    # ------------------------------------------------------------------
    @app.get("/api/config")
    async def api_get_config():
        """Returns all tunable parameters with current values + schema."""
        cfg = Config.get()
        return {
            "params": cfg.get_tunable_view(),
            "total": len(TUNABLE_SCHEMA),
        }

    @app.post("/api/config")
    async def api_update_config(req: ConfigUpdateRequest, token: Optional[str] = Query(None)):
        """Batch update tunable parameters. Hot-reloads live modules."""
        _check_token(token)
        if not req.updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        if len(req.updates) > 50:
            raise HTTPException(status_code=400, detail="Too many updates (max 50 per batch)")

        reloader = _get_hot_reloader()
        results = reloader.apply_updates(req.updates, source="dashboard")
        return {
            "results": [
                {"path": p, "success": ok, "message": msg}
                for p, ok, msg in results
            ],
            "total": len(results),
            "succeeded": sum(1 for _, ok, _ in results if ok),
            "failed": sum(1 for _, ok, _ in results if not ok),
        }

    @app.post("/api/config/reset")
    async def api_reset_config(path: str, token: Optional[str] = Query(None)):
        """Reset a single parameter to its current YAML value (no-op if unchanged)."""
        _check_token(token)
        # For reset, we just re-read the current YAML and reload.
        # This is essentially: discard in-memory change and re-apply persisted value.
        # Since we always persist on update, "reset" here means: do nothing
        # (the in-memory value already matches the persisted one).
        # To support true reset-to-default, we'd need to keep the example yaml.
        return {"message": "Reset not yet implemented; edit config.yaml manually and restart."}

    @app.get("/api/config/audit")
    async def api_config_audit(limit: int = Query(50, le=500)):
        reloader = _get_hot_reloader()
        return reloader.load_audit_log(limit=limit)

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------
    @app.post("/api/kill")
    async def api_kill(token: Optional[str] = Query(None)):
        _check_token(token)
        KillSwitch.trigger("Manual kill via dashboard")
        return {"status": "killed", "reason": KillSwitch.reason()}

    # ------------------------------------------------------------------
    # Analysis routes (advanced analytics)
    # ------------------------------------------------------------------
    @app.get("/api/analyze/{mint}")
    async def api_analyze_token(mint: str):
        """Run all analyzers on a token and return the composite AlphaSignal."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        try:
            signal = await orch.alpha_signal.generate(mint)
            return {
                "mint": signal.mint,
                "chain": signal.chain,
                "alpha_score": signal.alpha_score,
                "conviction": signal.conviction,
                "recommendation": signal.recommendation,
                "recommended_size_sol": signal.recommended_size_sol,
                "recommended_strategy": signal.recommended_strategy,
                "rationale": signal.rationale,
                "risks": signal.risks,
                "component_scores": signal.component_scores,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/order_flow/{mint}")
    async def api_order_flow(mint: str):
        """Get order flow snapshots for a token across multiple time windows."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        snapshots = orch.order_flow.get_multi_window(mint)
        out = {}
        for window, snap in snapshots.items():
            if snap:
                out[f"{window}s"] = {
                    "total_trades": snap.total_trades,
                    "buy_count": snap.buy_count,
                    "sell_count": snap.sell_count,
                    "buy_sell_ratio": snap.buy_sell_ratio,
                    "net_sol_flow": snap.net_sol_flow,
                    "volume_sol": snap.volume_sol,
                    "whale_buy_count": snap.whale_buy_count,
                    "whale_sell_count": snap.whale_sell_count,
                    "whale_net_sol": snap.whale_net_sol,
                    "unique_traders": snap.unique_traders,
                    "avg_trade_size_sol": snap.avg_trade_size_sol,
                    "trades_per_minute": snap.trades_per_minute,
                    "pressure_score": snap.pressure_score,
                }
            else:
                out[f"{window}s"] = None
        return out

    @app.get("/api/lifecycle/{mint}")
    async def api_lifecycle(mint: str):
        """Get lifecycle assessment for a token."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        try:
            assessment = await orch.lifecycle.assess(mint)
            return {
                "stage": assessment.stage.value,
                "age_minutes": assessment.age_minutes,
                "bonding_curve_pct": assessment.bonding_curve_pct,
                "holder_velocity": assessment.holder_velocity,
                "price_velocity_pct": assessment.price_velocity_pct,
                "trade_velocity": assessment.trade_velocity,
                "is_dying": assessment.is_dying,
                "is_breaking_out": assessment.is_breaking_out,
                "size_multiplier": assessment.size_multiplier,
                "recommended_strategy": assessment.recommended_strategy,
                "risk_level": assessment.risk_level,
                "notes": assessment.notes,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/liquidity/{mint}")
    async def api_liquidity(mint: str):
        """Get liquidity depth analysis for a token."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        try:
            depth = await orch.liquidity_depth.analyze(mint)
            if not depth:
                return {"error": "Could not fetch liquidity data"}
            return {
                "pool_type": depth.pool_type,
                "liquidity_usd": depth.liquidity_usd,
                "liquidity_sol": depth.liquidity_sol,
                "market_cap_usd": depth.market_cap_usd,
                "liquidity_to_mcap_ratio": depth.liquidity_to_mcap_ratio,
                "slippage_curve": depth.slippage_curve,
                "max_profitable_size_sol": depth.max_profitable_size_sol,
                "recommended_max_position_sol": depth.recommended_max_position_sol,
                "notes": depth.notes,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sentiment/{mint}")
    async def api_sentiment(mint: str):
        """Get sentiment snapshot for a token."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        snap = orch.sentiment.snapshot(mint)
        return {
            "mention_count_1h": snap.mention_count_1h,
            "mention_count_6h": snap.mention_count_6h,
            "mention_count_24h": snap.mention_count_24h,
            "sentiment_score": snap.sentiment_score,
            "mentions_per_minute": snap.mentions_per_minute,
            "influencer_mentions": snap.influencer_mentions,
            "hype_score": snap.hype_score,
            "top_keywords": snap.top_keywords,
        }

    @app.get("/api/mev/stats")
    async def api_mev_stats():
        """Get MEV detector stats."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        return orch.mev_detector.get_stats()

    @app.get("/api/wallet/{address}")
    async def api_wallet_profile(address: str):
        """Get on-chain reputation profile for a wallet."""
        orch = _orchestrator()
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not running")
        prof = orch.social_graph.get_wallet_profile(address)
        if not prof:
            return {"address": address, "known": False}
        return {
            "address": prof.address,
            "known": True,
            "first_seen_ts": prof.first_seen_ts,
            "total_trades": prof.total_trades,
            "profitable_trades": prof.profitable_trades,
            "total_pnl_usd": prof.total_pnl_usd,
            "avg_buy_to_peak_pct": prof.avg_buy_to_peak_pct,
            "early_buy_count": prof.early_buy_count,
            "tags": prof.tags,
            "reputation_score": prof.reputation_score,
            "cluster": orch.social_graph.get_cluster_for_wallet(address),
        }

    return app


# ----------------------------------------------------------------------
# Standalone launcher
# ----------------------------------------------------------------------
def run(port: int = 8080) -> None:
    import uvicorn
    app = build_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run()
