"""
Traffic AI Command Center – http://localhost:8000/
Aggregates all YOLO camera streams, serves a full-featured dashboard
with accident prediction, real-time alerts, scenario control,
multi-horizon traffic forecasting, signal recommendations, and
historical performance analysis.

Access control: login required (admin / traffic2024  or  operator / watch9x)
"""

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout

try:
    from storage import (get_recent_events as _db_events,
                         get_recent_forecasts as _db_forecasts,
                         get_performance_history as _db_perf,
                         get_event_counts_by_type as _db_counts,
                         save_recommendation as _db_save_rec)
    _STORAGE_OK = True
except ImportError:
    _STORAGE_OK = False
    def _db_events(limit=100): return []
    def _db_forecasts(limit=50): return []
    def _db_perf(cam, n=60): return []
    def _db_counts(hours=24): return {}
    def _db_save_rec(r): pass

FORECAST_URL = "http://127.0.0.1:8090"

# ── Auth ───────────────────────────────────────────────────────────────────────
USERS: dict = {"admin": "traffic2024", "operator": "watch9x"}
SESSIONS: dict = {}   # token → username

def _check_session(request) -> str | None:
    token = request.cookies.get("session")
    return SESSIONS.get(token)

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic AI – Login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:#07090e;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.card{background:#0f1520;border:1px solid #1e2d45;border-radius:14px;padding:40px 36px;
  width:min(90vw,360px);box-shadow:0 24px 80px rgba(0,0,0,.6)}
h1{font-size:18px;font-weight:700;color:#fff;margin-bottom:4px}
p{font-size:12px;color:#64748b;margin-bottom:28px}
label{display:block;font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:6px}
input{width:100%;padding:10px 12px;background:#161e2e;border:1px solid #1e2d45;
  border-radius:7px;color:#e2e8f0;font-size:13px;margin-bottom:16px;outline:none}
input:focus{border-color:#3b82f6}
button{width:100%;padding:11px;background:#3b82f6;border:none;border-radius:7px;
  color:#fff;font-size:13px;font-weight:700;cursor:pointer;transition:background .15s}
button:hover{background:#2563eb}
.err{color:#fca5a5;font-size:12px;margin-top:10px;text-align:center}
.logo{font-size:28px;text-align:center;margin-bottom:20px}
</style>
</head>
<body>
<div class="card">
  <div class="logo">🚦</div>
  <h1>Traffic AI Command Center</h1>
  <p>Authorized access only. Enter your credentials.</p>
  <form method="POST" action="/login">
    <label>Username</label>
    <input type="text" name="username" placeholder="username" autocomplete="username" required>
    <label>Password</label>
    <input type="password" name="password" placeholder="••••••••" autocomplete="current-password" required>
    <button type="submit">Sign In</button>
    ERROR_MSG
  </form>
</div>
</body></html>
"""

CAMERAS = {
    "north": {"label": "CAM-01 NORTH – FLOW MONITOR",     "url": "http://127.0.0.1:8011"},
    "south": {"label": "CAM-02 SOUTH – STOP & STALL",     "url": "http://127.0.0.1:8012"},
    "east":  {"label": "CAM-03 EAST – FULL MONITOR",      "url": "http://127.0.0.1:8013"},
    "west":  {"label": "CAM-04 WEST – ACCIDENT DETECTION","url": "http://127.0.0.1:8014"},
}

SCENARIO_LABELS = {
    "normal":       "Normal Flow",
    "rush_hour":    "Rush Hour",
    "accident_zone":"Accident Zone",
    "heavy_rain":   "Heavy Rain",
    "night":        "Night Traffic",
}

# ── Embedded HTML dashboard ────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic AI Command Center</title>
<style>
:root{
  --bg:#07090e;--surface:#0f1520;--surface2:#161e2e;--border:#1e2d45;
  --accent:#3b82f6;--green:#22c55e;--yellow:#eab308;--red:#ef4444;
  --text:#e2e8f0;--muted:#64748b;--radius:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px}
body{display:flex;flex-direction:column;min-height:100vh}

/* ── Header ── */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 20px;background:var(--surface);border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{font-size:22px}
.brand h1{font-size:15px;font-weight:700;letter-spacing:.04em;color:#fff}
.brand p{font-size:11px;color:var(--muted)}
.header-right{display:flex;align-items:center;gap:12px}
.live-badge{
  display:flex;align-items:center;gap:5px;padding:4px 10px;
  background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);
  border-radius:999px;font-size:11px;font-weight:600;color:var(--green)
}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.clock{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}

/* ── Accident Alert Banner ── */
#alert-banner{
  display:none;align-items:center;justify-content:space-between;
  padding:12px 20px;background:linear-gradient(90deg,#7f1d1d,#991b1b);
  border-bottom:2px solid var(--red);animation:blink-border 0.6s infinite alternate;
}
@keyframes blink-border{from{border-color:#ef4444}to{border-color:#fca5a5}}
#alert-banner.active{display:flex}
.alert-text{display:flex;align-items:center;gap:10px;font-weight:700;font-size:14px;color:#fef2f2}
.alert-icon{font-size:20px;animation:shake .4s infinite}
@keyframes shake{0%,100%{transform:rotate(0)}25%{transform:rotate(-6deg)}75%{transform:rotate(6deg)}}
.alert-close{background:none;border:none;color:#fca5a5;font-size:18px;cursor:pointer;padding:0 4px}

/* ── Main layout ── */
main{flex:1;display:grid;grid-template-columns:1fr 320px;gap:16px;padding:16px;overflow:hidden}

/* ── Camera grid ── */
.camera-section h2{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
.camera-grid{
  display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
}
.cam-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  overflow:hidden;position:relative;transition:border-color .2s;
}
.cam-card.alert-cam{border-color:var(--red);box-shadow:0 0 12px rgba(239,68,68,.3)}
.cam-feed{width:100%;aspect-ratio:16/9;background:#000;display:block;object-fit:cover}
.cam-feed.offline{background:linear-gradient(135deg,#0f1520 0%,#161e2e 100%);min-height:100px}
.cam-overlay{
  position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(to top,rgba(0,0,0,.85),transparent);
  padding:8px 10px 6px;
}
.cam-name{font-size:11px;font-weight:700;color:#fff;letter-spacing:.04em}
.cam-risk{
  display:inline-flex;align-items:center;gap:4px;
  font-size:10px;font-weight:700;margin-top:2px;padding:2px 7px;
  border-radius:999px;
}
.cam-risk.low{background:rgba(34,197,94,.2);color:#86efac}
.cam-risk.med{background:rgba(234,179,8,.2);color:#fde047}
.cam-risk.high{background:rgba(239,68,68,.25);color:#fca5a5;animation:pulse .8s infinite}
.cam-counts{font-size:10px;color:rgba(255,255,255,.7);margin-top:2px}
.cam-badge-offline{
  position:absolute;top:8px;right:8px;background:rgba(0,0,0,.7);
  border:1px solid var(--border);border-radius:4px;
  font-size:9px;color:var(--muted);padding:2px 6px;
}

/* ── Right sidebar ── */
.sidebar{display:flex;flex-direction:column;gap:12px;overflow-y:auto;max-height:calc(100vh - 180px)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px}
.panel-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;display:flex;align-items:center;gap:6px}

/* Risk panel */
.risk-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.risk-label{font-size:11px;color:var(--text);width:120px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.risk-bar-wrap{flex:1;background:var(--surface2);border-radius:999px;height:8px;overflow:hidden}
.risk-bar-fill{height:100%;border-radius:999px;transition:width .4s,background .4s}
.risk-pct{font-size:11px;font-variant-numeric:tabular-nums;width:34px;text-align:right}

/* Stats panel */
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat-card{background:var(--surface2);border-radius:8px;padding:10px;text-align:center}
.stat-val{font-size:22px;font-weight:800;color:#fff;font-variant-numeric:tabular-nums}
.stat-lbl{font-size:10px;color:var(--muted);margin-top:2px}

/* Scenarios panel */
.scenario-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.scenario-btn{
  padding:8px 6px;border-radius:7px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);font-size:11px;font-weight:600;
  cursor:pointer;transition:all .15s;text-align:center;
}
.scenario-btn:hover{border-color:var(--accent);color:#fff;background:rgba(59,130,246,.12)}
.scenario-btn.active{border-color:var(--accent);background:rgba(59,130,246,.18);color:#93c5fd}
.scenario-btn.danger{border-color:rgba(239,68,68,.5);color:#fca5a5}
.scenario-btn.danger.active{background:rgba(239,68,68,.18);border-color:var(--red)}

/* Events panel */
.events-list{display:flex;flex-direction:column;gap:5px;max-height:280px;overflow-y:auto}
.event-item{
  display:flex;align-items:flex-start;gap:8px;padding:7px 9px;
  border-radius:7px;border-left:3px solid transparent;background:var(--surface2);
  font-size:11px;
}
.event-item.accident{border-left-color:var(--red)}
.event-item.warning{border-left-color:var(--yellow)}
.event-item.info{border-left-color:var(--accent)}
.event-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:3px}
.event-dot.red{background:var(--red)}
.event-dot.yellow{background:var(--yellow)}
.event-dot.blue{background:var(--accent)}
.event-dot.green{background:var(--green)}
.event-body{flex:1;min-width:0}
.event-type{font-weight:700;color:#fff;text-transform:capitalize}
.event-meta{color:var(--muted);margin-top:1px}

/* Health panel */
.health-row{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;border-bottom:1px solid var(--border)}
.health-row:last-child{border:none}
.health-val{color:var(--green);font-weight:700;font-variant-numeric:tabular-nums}
.health-val.warn{color:var(--yellow)}
.health-val.err{color:var(--red)}

/* No events placeholder */
.no-events{color:var(--muted);font-size:12px;text-align:center;padding:20px 0}

/* Scrollbar */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

@media(max-width:1100px){
  main{grid-template-columns:1fr}
  .camera-grid{grid-template-columns:repeat(2,1fr)}
  .sidebar{max-height:none}
}
@media(max-width:700px){
  .camera-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-icon">🚦</span>
    <div>
      <h1>Traffic AI Command Center</h1>
      <p>YOLOv12 · Multi-Camera · Accident Prediction</p>
    </div>
  </div>
  <div class="header-right">
    <div class="live-badge"><span class="live-dot"></span>LIVE</div>
    <div class="clock" id="clock">--:--:--</div>
  </div>
</header>

<div id="alert-banner">
  <div class="alert-text">
    <span class="alert-icon">🚨</span>
    <span id="alert-msg">ACCIDENT DETECTED – Immediate attention required</span>
  </div>
  <button class="alert-close" onclick="dismissAlert()">✕</button>
</div>

<main>
  <!-- Camera grid -->
  <section class="camera-section">
    <h2>Live Camera Feeds</h2>
    <div class="camera-grid" id="camera-grid">
      <!-- populated by JS -->
    </div>
  </section>

  <!-- Sidebar -->
  <aside class="sidebar">

    <!-- Accident Risk -->
    <div class="panel">
      <div class="panel-title">⚠️ Accident Risk</div>
      <div id="risk-panel">
        <!-- populated by JS -->
      </div>
    </div>

    <!-- Vehicle Totals -->
    <div class="panel">
      <div class="panel-title">🚗 Vehicle Counts</div>
      <div class="stats-grid" id="count-stats">
        <div class="stat-card"><div class="stat-val" id="cnt-car">0</div><div class="stat-lbl">Cars</div></div>
        <div class="stat-card"><div class="stat-val" id="cnt-truck">0</div><div class="stat-lbl">Trucks</div></div>
        <div class="stat-card"><div class="stat-val" id="cnt-bus">0</div><div class="stat-lbl">Buses</div></div>
        <div class="stat-card"><div class="stat-val" id="cnt-person">0</div><div class="stat-lbl">Pedestrians</div></div>
        <div class="stat-card"><div class="stat-val" id="cnt-moto">0</div><div class="stat-lbl">Motos</div></div>
        <div class="stat-card"><div class="stat-val" id="cnt-total">0</div><div class="stat-lbl">Total</div></div>
      </div>
    </div>

    <!-- Scenarios -->
    <div class="panel">
      <div class="panel-title">🎬 Scenarios</div>
      <div class="scenario-grid">
        <button class="scenario-btn active" data-scenario="normal" onclick="setScenario(this)">Normal Flow</button>
        <button class="scenario-btn" data-scenario="rush_hour" onclick="setScenario(this)">Rush Hour</button>
        <button class="scenario-btn danger" data-scenario="accident_zone" onclick="setScenario(this)">Accident Zone</button>
        <button class="scenario-btn" data-scenario="heavy_rain" onclick="setScenario(this)">Heavy Rain</button>
        <button class="scenario-btn" data-scenario="night" onclick="setScenario(this)">Night Traffic</button>
      </div>
    </div>

    <!-- Live Events -->
    <div class="panel">
      <div class="panel-title">📋 Live Events</div>
      <div class="events-list" id="events-list">
        <div class="no-events">Waiting for events…</div>
      </div>
    </div>

    <!-- System Health -->
    <div class="panel">
      <div class="panel-title">💚 System Health</div>
      <div id="health-panel">
        <div class="health-row"><span>Active Cameras</span><span class="health-val" id="h-cams">0/5</span></div>
        <div class="health-row"><span>Frames Processed</span><span class="health-val" id="h-frames">0</span></div>
        <div class="health-row"><span>Frames Dropped</span><span class="health-val" id="h-dropped">0</span></div>
        <div class="health-row"><span>Uptime</span><span class="health-val" id="h-uptime">0s</span></div>
        <div class="health-row"><span>Reconnects</span><span class="health-val" id="h-reconnects">0</span></div>
      </div>
    </div>

    <!-- Traffic Forecasts -->
    <div class="panel">
      <div class="panel-title">📈 Traffic Forecast</div>
      <div id="forecast-panel">
        <div class="no-events" style="font-size:11px">Loading forecasts…</div>
      </div>
    </div>

    <!-- Signal Recommendations -->
    <div class="panel">
      <div class="panel-title">🚦 Signal Recommendations</div>
      <div id="signal-recs-panel">
        <div class="no-events" style="font-size:11px">Loading recommendations…</div>
      </div>
    </div>

    <!-- Historical Event Summary -->
    <div class="panel">
      <div class="panel-title">📊 24h Event Summary</div>
      <div id="historical-panel">
        <div class="no-events" style="font-size:11px">Loading history…</div>
      </div>
    </div>

    <!-- Logout -->
    <div style="text-align:center;padding:4px 0 8px">
      <a href="/logout" style="font-size:11px;color:#64748b;text-decoration:none">Sign out</a>
    </div>

  </aside>
</main>

<script>
// ── State ────────────────────────────────────────────────────────────────────
const CAMS = CAMERA_CONFIG_JSON;
const CAM_KEYS = Object.keys(CAMS);
let allStats = {};
let allEvents = [];
let dismissedAt = 0;
let currentScenario = 'normal';
let alertActive = false;

// ── Clock ───────────────────────────────────────────────────────────────────
function updateClock(){
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('en-GB', {hour12: false});
}
setInterval(updateClock, 1000);
updateClock();

// ── Build camera grid ────────────────────────────────────────────────────────
function buildCameraGrid(){
  const grid = document.getElementById('camera-grid');
  grid.innerHTML = '';
  CAM_KEYS.forEach(key => {
    const cam = CAMS[key];
    grid.innerHTML += `
      <div class="cam-card" id="card-${key}">
        <img class="cam-feed" id="feed-${key}"
          src="/feed/${key}"
          alt="${cam.label}"
          onerror="handleFeedError('${key}')"
          onload="handleFeedLoad('${key}')"
        >
        <div class="cam-badge-offline" id="offline-${key}" style="display:none">OFFLINE</div>
        <div class="cam-overlay">
          <div class="cam-name">${cam.label}</div>
          <div class="cam-risk low" id="risk-badge-${key}">RISK 0%</div>
          <div class="cam-counts" id="counts-${key}">No data</div>
        </div>
      </div>`;
  });
}

function handleFeedError(key){
  const badge = document.getElementById(`offline-${key}`);
  if(badge) badge.style.display='block';
  const feed = document.getElementById(`feed-${key}`);
  if(feed){ feed.classList.add('offline'); feed.removeAttribute('src'); }
}

function handleFeedLoad(key){
  const badge = document.getElementById(`offline-${key}`);
  if(badge) badge.style.display='none';
  const feed = document.getElementById(`feed-${key}`);
  if(feed) feed.classList.remove('offline');
}

// ── Sound alert (Web Audio API) ──────────────────────────────────────────────
let audioCtx = null;
function getAudio(){ if(!audioCtx) audioCtx = new AudioContext(); return audioCtx; }

function playAccidentAlert(){
  try{
    const ctx = getAudio();
    [0, 0.25, 0.5].forEach(delay => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.frequency.setValueAtTime(880, ctx.currentTime + delay);
      osc.frequency.setValueAtTime(660, ctx.currentTime + delay + 0.12);
      gain.gain.setValueAtTime(0.6, ctx.currentTime + delay);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + 0.22);
      osc.start(ctx.currentTime + delay);
      osc.stop(ctx.currentTime + delay + 0.24);
    });
  } catch(_){}
}

function playWarningBeep(){
  try{
    const ctx = getAudio();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = 520;
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);
    osc.start(); osc.stop(ctx.currentTime + 0.2);
  } catch(_){}
}

// ── Alert banner ─────────────────────────────────────────────────────────────
function showAlert(msg, cameras){
  if(alertActive) return;
  const banner = document.getElementById('alert-banner');
  document.getElementById('alert-msg').textContent = msg;
  banner.classList.add('active');
  alertActive = true;
  playAccidentAlert();
  cameras.forEach(k => {
    const card = document.getElementById(`card-${k}`);
    if(card) card.classList.add('alert-cam');
  });
}

function dismissAlert(){
  const banner = document.getElementById('alert-banner');
  banner.classList.remove('active');
  alertActive = false;
  dismissedAt = Date.now();
  CAM_KEYS.forEach(k => {
    const card = document.getElementById(`card-${k}`);
    if(card) card.classList.remove('alert-cam');
  });
}

// ── Risk colour helper ────────────────────────────────────────────────────────
function riskClass(r){ return r >= 75 ? 'high' : r >= 45 ? 'med' : 'low'; }
function riskBarColor(r){
  if(r >= 75) return '#ef4444';
  if(r >= 45) return '#eab308';
  return '#22c55e';
}

// ── Update UI from stats ──────────────────────────────────────────────────────
function updateFromStats(stats){
  allStats = stats;

  // Per-camera updates
  let activeCams = 0;
  let accidentCams = [];
  const riskPanel = document.getElementById('risk-panel');
  riskPanel.innerHTML = '';

  CAM_KEYS.forEach(key => {
    const s = stats[key];
    if(!s) return;
    activeCams++;

    const risk = s.accident_risk || 0;
    const badge = document.getElementById(`risk-badge-${key}`);
    if(badge){
      badge.textContent = `RISK ${risk.toFixed(0)}%`;
      badge.className = `cam-risk ${riskClass(risk)}`;
    }
    const countEl = document.getElementById(`counts-${key}`);
    if(countEl && s.counts){
      const c = s.counts;
      countEl.textContent = `Cars:${c.car||0} Trucks:${c.truck||0} People:${c.person||0} Total:${c.total||0}`;
    }
    if(s.accident_alert){ accidentCams.push(key); }

    // Risk bar row
    riskPanel.innerHTML += `
      <div class="risk-row">
        <span class="risk-label">${s.camera_id || key}</span>
        <div class="risk-bar-wrap">
          <div class="risk-bar-fill" style="width:${risk}%;background:${riskBarColor(risk)}"></div>
        </div>
        <span class="risk-pct" style="color:${riskBarColor(risk)}">${risk.toFixed(0)}%</span>
      </div>`;
  });

  // Aggregate vehicle counts
  let totals = {car:0,truck:0,bus:0,person:0,motorcycle:0,total:0};
  Object.values(stats).forEach(s => {
    if(!s || !s.counts) return;
    Object.keys(totals).forEach(k => { totals[k] += (s.counts[k]||0); });
  });
  document.getElementById('cnt-car').textContent   = totals.car;
  document.getElementById('cnt-truck').textContent = totals.truck;
  document.getElementById('cnt-bus').textContent   = totals.bus;
  document.getElementById('cnt-person').textContent= totals.person;
  document.getElementById('cnt-moto').textContent  = totals.motorcycle;
  document.getElementById('cnt-total').textContent = totals.total;

  // Health
  let fp=0,fd=0,up=0,rc=0;
  Object.values(stats).forEach(s => {
    if(!s) return;
    fp += s.frames_processed||0;
    fd += s.frames_dropped||0;
    up = Math.max(up, s.stream_uptime_s||0);
    rc += s.reconnect_count||0;
  });
  document.getElementById('h-cams').textContent      = `${activeCams}/${CAM_KEYS.length}`;
  document.getElementById('h-frames').textContent    = fp.toLocaleString();
  document.getElementById('h-dropped').textContent   = fd;
  document.getElementById('h-uptime').textContent    = formatUptime(up);
  document.getElementById('h-reconnects').textContent= rc;
  document.getElementById('h-dropped').className     = `health-val ${fd > 100 ? 'warn' : ''}`;

  // Accident alert
  if(accidentCams.length > 0 && Date.now() - dismissedAt > 8000){
    const camNames = accidentCams.map(k => CAMS[k]?.label || k).join(', ');
    showAlert(`🚨 ACCIDENT DETECTED · ${camNames}`, accidentCams);
  }
}

function formatUptime(s){
  if(s < 60) return `${s}s`;
  if(s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}

// ── Update event feed ─────────────────────────────────────────────────────────
const EVENT_ICONS = {
  accident_detected:'🔴', demo_accident:'🔴',
  stalled_vehicle:'🟡', abnormal_stopping:'🟡',
  queue_spillback:'🟠', sudden_congestion:'🟠',
  unexpected_trajectory:'🔵',
};
const EVENT_CLASS = {
  accident_detected:'accident', demo_accident:'accident',
  stalled_vehicle:'warning', abnormal_stopping:'warning',
  queue_spillback:'warning', sudden_congestion:'warning',
  unexpected_trajectory:'info',
};
const DOT_CLASS = {
  accident_detected:'red', demo_accident:'red',
  stalled_vehicle:'yellow', abnormal_stopping:'yellow',
  queue_spillback:'yellow', sudden_congestion:'yellow',
  unexpected_trajectory:'blue',
};

let lastEventCount = 0;
function updateEvents(events){
  if(events.length === lastEventCount) return;

  const hasNew = events.length > lastEventCount;
  const newAccident = events.slice(0, events.length - lastEventCount)
    .some(e => e.event_type === 'accident_detected' || e.event_type === 'demo_accident');

  lastEventCount = events.length;
  allEvents = events;

  if(hasNew && !newAccident) playWarningBeep();

  const list = document.getElementById('events-list');
  if(!events.length){
    list.innerHTML = '<div class="no-events">No events yet</div>';
    return;
  }
  list.innerHTML = events.slice(0, 40).map(ev => {
    const icon  = EVENT_ICONS[ev.event_type]  || '⚪';
    const cls   = EVENT_CLASS[ev.event_type]  || 'info';
    const dot   = DOT_CLASS[ev.event_type]    || 'blue';
    const conf  = ev.confidence ? `${(ev.confidence*100).toFixed(0)}%` : '';
    const ts    = ev.timestamp ? ev.timestamp.slice(11,19) : '';
    const typ   = (ev.event_type||'').replace(/_/g,' ');
    return `<div class="event-item ${cls}">
      <span class="event-dot ${dot}"></span>
      <div class="event-body">
        <div class="event-type">${icon} ${typ} ${conf}</div>
        <div class="event-meta">${ts} · ${ev.camera_id||''} · ${(ev.location||'').split('·').pop().trim()}</div>
      </div>
    </div>`;
  }).join('');
}

// ── Scenario control ──────────────────────────────────────────────────────────
async function setScenario(btn){
  const scenario = btn.dataset.scenario;
  currentScenario = scenario;
  document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  try{
    const results = await Promise.all(CAM_KEYS.map(key => {
      const base = CAMS[key].url;
      return fetch(`/proxy_scenario/${key}`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({scenario})
      }).catch(()=>{});
    }));
    console.log(`Scenario set: ${scenario}`);
  } catch(e){ console.warn('Scenario error:', e); }

  // Visual effects per scenario
  applyScenarioVisual(scenario);
}

function applyScenarioVisual(scenario){
  const body = document.body;
  body.classList.remove('scenario-night','scenario-rain','scenario-rush');
  if(scenario==='night') body.classList.add('scenario-night');
  if(scenario==='heavy_rain') body.classList.add('scenario-rain');
  if(scenario==='rush_hour') body.classList.add('scenario-rush');
}

// ── Data polling ──────────────────────────────────────────────────────────────
async function fetchStats(){
  try{
    const r = await fetch('/api/stats');
    if(r.ok){ const data = await r.json(); updateFromStats(data); }
  } catch(_){}
}

async function fetchEvents(){
  try{
    const r = await fetch('/api/events');
    if(r.ok){ const data = await r.json(); updateEvents(data); }
  } catch(_){}
}

async function poll(){
  await Promise.all([fetchStats(), fetchEvents()]);
  setTimeout(poll, 1200);
}

// ── WebSocket (real-time push) ────────────────────────────────────────────────
function connectWS(){
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = e => {
    try{
      const msg = JSON.parse(e.data);
      if(msg.type === 'stats')  updateFromStats(msg.data);
      if(msg.type === 'events') updateEvents(msg.data);
    } catch(_){}
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ── Traffic Forecast ─────────────────────────────────────────────────────────
async function fetchForecast(){
  try{
    // Build request from current aggregated stats
    let totalCount = 0;
    Object.values(allStats).forEach(s => { if(s && s.counts) totalCount += (s.counts.total||0); });
    const now = new Date();
    const body = {
      hour: now.getHours() + now.getMinutes()/60,
      dayofweek: now.getDay() === 0 ? 6 : now.getDay() - 1,
      is_weekend: [5,6].includes(now.getDay()) ? 1 : 0,
      temperature: 28, rain: 0,
      vehicle_count_last_15min: Math.max(5, totalCount),
      dominant_phase: 1
    };
    const r = await fetch('/api/forecast', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    if(!r.ok) return;
    const data = await r.json();
    updateForecastPanel(data);
    updateSignalPanel(data.recommendations || []);
  } catch(_){}
}

function horizonColor(v, base){
  const ratio = base > 0 ? v/base : 1;
  if(ratio >= 1.4) return '#ef4444';
  if(ratio >= 1.15) return '#eab308';
  return '#22c55e';
}

function updateForecastPanel(data){
  const panel = document.getElementById('forecast-panel');
  if(!panel) return;
  const total = data.total || {};
  const base  = total.horizon_15 || 1;
  const model = total.model || 'rf';
  const pa    = data.per_approach || {};

  let html = `<div style="font-size:10px;color:#64748b;margin-bottom:8px">Model: ${model.toUpperCase()} · Wadi Saqra</div>`;
  // Horizon cards
  html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:12px">`;
  [{label:'+15 min',val:total.horizon_15},{label:'+30 min',val:total.horizon_30},{label:'+60 min',val:total.horizon_60}].forEach(h=>{
    const c = horizonColor(h.val, total.horizon_15||1);
    html += `<div style="background:#161e2e;border-radius:7px;padding:8px;text-align:center">
      <div style="font-size:16px;font-weight:800;color:${c}">${(h.val||0).toFixed(0)}</div>
      <div style="font-size:10px;color:#64748b">${h.label}</div>
    </div>`;
  });
  html += `</div>`;
  // Per-approach rows
  ['North','South','East','West'].forEach(ap => {
    const v = pa[ap] || {};
    if(!v.horizon_15) return;
    const c = horizonColor(v.horizon_15, v.current||1);
    html += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;font-size:11px">
      <span style="width:50px;color:#94a3b8">${ap}</span>
      <div style="flex:1;background:#161e2e;border-radius:4px;height:6px;overflow:hidden">
        <div style="height:100%;background:${c};width:${Math.min(100,(v.horizon_15/80)*100).toFixed(0)}%;border-radius:4px"></div>
      </div>
      <span style="color:${c};font-variant-numeric:tabular-nums;width:34px;text-align:right">${(v.horizon_15||0).toFixed(0)}</span>
    </div>`;
  });
  panel.innerHTML = html;
}

// ── Signal Recommendations ────────────────────────────────────────────────────
const REC_COLOR = {EXTEND_GREEN:'#22c55e', REDUCE_GREEN:'#eab308', MAINTAIN:'#64748b'};
const REC_ICON  = {EXTEND_GREEN:'▲', REDUCE_GREEN:'▼', MAINTAIN:'─'};
const URG_COLOR = {HIGH:'#ef4444', MEDIUM:'#eab308', LOW:'#64748b', NONE:'#1e2d45'};

function updateSignalPanel(recs){
  const panel = document.getElementById('signal-recs-panel');
  if(!panel || !recs.length){ if(panel) panel.innerHTML='<div class="no-events" style="font-size:11px">No recommendations</div>'; return; }
  panel.innerHTML = recs.map(r => `
    <div style="background:#161e2e;border-radius:7px;padding:8px 10px;margin-bottom:6px;border-left:3px solid ${URG_COLOR[r.urgency]||'#1e2d45'}">
      <div style="display:flex;justify-content:space-between;margin-bottom:2px">
        <span style="font-weight:700;color:#fff;font-size:11px">${r.approach}</span>
        <span style="font-size:11px;color:${REC_COLOR[r.action]||'#fff'}">${REC_ICON[r.action]||''} ${r.action.replace(/_/g,' ')}</span>
      </div>
      <div style="font-size:10px;color:#64748b">${r.current_green_s}s → ${r.recommended_green_s}s · ${r.reason||''}</div>
    </div>`).join('');
}

// ── Historical summary ────────────────────────────────────────────────────────
async function fetchHistorical(){
  try{
    const r = await fetch('/api/historical');
    if(r.ok){ updateHistoricalPanel(await r.json()); }
  } catch(_){}
}

function updateHistoricalPanel(data){
  const panel = document.getElementById('historical-panel');
  if(!panel) return;
  const types = data.event_counts || {};
  if(!Object.keys(types).length){ panel.innerHTML='<div class="no-events" style="font-size:11px">No history yet</div>'; return; }
  const TYPE_COLOR = {
    accident_detected:'#ef4444', demo_accident:'#ef4444',
    stalled_vehicle:'#eab308', abnormal_stopping:'#eab308',
    queue_spillback:'#f97316', sudden_congestion:'#f97316',
    unexpected_trajectory:'#3b82f6'
  };
  const max = Math.max(...Object.values(types));
  panel.innerHTML = Object.entries(types).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([k,v])=>{
    const c = TYPE_COLOR[k]||'#64748b';
    const pct = Math.max(4, (v/max)*100).toFixed(0);
    return `<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">
      <span style="font-size:10px;color:#94a3b8;width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${k.replace(/_/g,' ')}</span>
      <div style="flex:1;background:#161e2e;border-radius:4px;height:6px;overflow:hidden">
        <div style="height:100%;background:${c};width:${pct}%;border-radius:4px"></div>
      </div>
      <span style="color:${c};font-size:10px;font-variant-numeric:tabular-nums;width:24px;text-align:right">${v}</span>
    </div>`;
  }).join('');

  // Append total
  const total = Object.values(types).reduce((a,b)=>a+b,0);
  panel.innerHTML += `<div style="font-size:10px;color:#64748b;margin-top:6px;text-align:right">Total last 24h: ${total}</div>`;
}

// ── Forecast + recommendations polling ───────────────────────────────────────
async function pollForecasts(){
  await Promise.all([fetchForecast(), fetchHistorical()]);
  setTimeout(pollForecasts, 30000);  // every 30 s
}

// ── Boot ──────────────────────────────────────────────────────────────────────
buildCameraGrid();
poll();
connectWS();
pollForecasts();
</script>
</body>
</html>
"""


# ── Dashboard server ───────────────────────────────────────────────────────────

class DashboardServer:
    def __init__(self, cameras):
        self.cameras = cameras
        self._ws_clients: set = set()
        self._last_stats: dict = {}
        self._last_events: list = []

    async def poll_loop(self):
        timeout = ClientTimeout(total=3.0)
        while True:
            stats = {}
            events = []
            async with ClientSession(timeout=timeout) as session:
                for key, cam in self.cameras.items():
                    try:
                        async with session.get(cam["url"] + "/stats") as r:
                            s = await r.json()
                            stats[key] = s
                    except Exception:
                        stats[key] = None
                    try:
                        async with session.get(cam["url"] + "/events") as r:
                            evs = await r.json()
                            events.extend(evs)
                    except Exception:
                        pass

            events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
            seen = set()
            deduped = []
            for ev in events:
                key_ev = (ev.get("timestamp"), ev.get("event_type"), ev.get("camera_id"), ev.get("track_id"))
                if key_ev not in seen:
                    seen.add(key_ev)
                    deduped.append(ev)
            self._last_stats = stats
            self._last_events = deduped[:120]

            await self._broadcast({"type": "stats", "data": stats})
            await self._broadcast({"type": "events", "data": self._last_events})
            await asyncio.sleep(1.2)

    async def _broadcast(self, payload):
        if not self._ws_clients:
            return
        text = json.dumps(payload)
        dead = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_str(text)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead


def build_app(server: DashboardServer, cameras: dict) -> web.Application:

    # Inject camera config into HTML
    cam_json = json.dumps({k: {"label": v["label"], "url": v["url"]} for k, v in cameras.items()})
    dashboard_html = DASHBOARD_HTML.replace("CAMERA_CONFIG_JSON", cam_json)

    PUBLIC_PATHS = {"/login", "/login/submit"}

    @web.middleware
    async def auth_mw(request, handler):
        path = request.path
        if path not in PUBLIC_PATHS and not path.startswith("/feed/"):
            if not _check_session(request):
                raise web.HTTPFound("/login")
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    async def login_get(_req):
        html = LOGIN_HTML.replace("ERROR_MSG", "")
        return web.Response(text=html, content_type="text/html")

    async def login_post(request):
        data = await request.post()
        user = data.get("username", "")
        pw   = data.get("password", "")
        if USERS.get(user) == pw:
            token = secrets.token_hex(24)
            SESSIONS[token] = user
            resp = web.HTTPFound("/")
            resp.set_cookie("session", token, httponly=True, samesite="Lax", max_age=86400)
            raise resp
        html = LOGIN_HTML.replace("ERROR_MSG", '<div class="err">Invalid credentials</div>')
        return web.Response(text=html, content_type="text/html")

    async def logout(request):
        token = request.cookies.get("session")
        if token:
            SESSIONS.pop(token, None)
        resp = web.HTTPFound("/login")
        resp.del_cookie("session")
        raise resp

    async def index(_req):
        return web.Response(text=dashboard_html, content_type="text/html")

    async def api_stats(_req):
        return web.json_response(server._last_stats)

    async def api_events(_req):
        return web.json_response(server._last_events)

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        server._ws_clients.add(ws)
        try:
            # Send current state immediately on connect
            await ws.send_str(json.dumps({"type": "stats", "data": server._last_stats}))
            await ws.send_str(json.dumps({"type": "events", "data": server._last_events}))
            async for _ in ws:
                pass
        finally:
            server._ws_clients.discard(ws)
        return ws

    async def feed_proxy(request):
        cam_key = request.match_info["camera"]
        cam = cameras.get(cam_key)
        if not cam:
            raise web.HTTPNotFound()
        upstream_url = cam["url"] + "/video_feed"
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )
        await response.prepare(request)
        timeout = ClientTimeout(total=None, connect=3.0)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.get(upstream_url) as upstream:
                    async for chunk in upstream.content.iter_chunked(131072):
                        await response.write(chunk)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            pass
        return response

    async def proxy_scenario(request):
        cam_key = request.match_info["camera"]
        cam = cameras.get(cam_key)
        if not cam:
            raise web.HTTPNotFound()
        body = await request.json()
        timeout = ClientTimeout(total=3.0)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.post(cam["url"] + "/scenario", json=body) as r:
                    result = await r.json()
                    return web.json_response(result)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=502)

    async def api_forecast(request):
        """Proxy multi-horizon forecast to the FastAPI service at port 8090."""
        body = await request.json()
        timeout = ClientTimeout(total=4.0)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.post(FORECAST_URL + "/predict/multihorizon", json=body) as r:
                    result = await r.json()
                    # Persist recommendations to DB
                    for rec in result.get("recommendations", []):
                        _db_save_rec(rec)
                    return web.json_response(result)
        except Exception as exc:
            return web.json_response({"error": str(exc), "total": {}, "per_approach": {}, "recommendations": []})

    async def api_historical(_req):
        """Return event type counts from the DB + recent forecast summaries."""
        counts = _db_counts(hours=24)
        forecasts = _db_forecasts(limit=10)
        return web.json_response({"event_counts": counts, "recent_forecasts": forecasts})

    async def api_db_events(_req):
        """Return persisted events from SQLite (richer than in-memory)."""
        events = _db_events(limit=100)
        return web.json_response(events)

    async def on_startup(app):
        app["poll_task"] = asyncio.create_task(server.poll_loop())

    async def on_cleanup(app):
        app["poll_task"].cancel()
        try:
            await app["poll_task"]
        except asyncio.CancelledError:
            pass

    app = web.Application(middlewares=[auth_mw])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/login",         login_get)
    app.router.add_post("/login",        login_post)
    app.router.add_get("/login/submit",  login_post)
    app.router.add_get("/logout",        logout)
    app.router.add_get("/",              index)
    app.router.add_get("/ws",            ws_handler)
    app.router.add_get("/api/stats",     api_stats)
    app.router.add_get("/api/events",    api_events)
    app.router.add_get("/api/events/db", api_db_events)
    app.router.add_post("/api/forecast", api_forecast)
    app.router.add_get("/api/historical",api_historical)
    app.router.add_get("/feed/{camera}", feed_proxy)
    app.router.add_post("/proxy_scenario/{camera}", proxy_scenario)
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic AI Dashboard – port 8000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    server = DashboardServer(CAMERAS)
    app = build_app(server, CAMERAS)
    print(f"Traffic AI Dashboard  →  http://localhost:{args.port}/", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
