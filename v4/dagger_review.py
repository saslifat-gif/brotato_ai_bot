"""Self-contained visual review page for DAgger corrective states.

The shadow evaluator records structured world state rather than screenshots.
This module turns those captured coordinates into a small offline tactical map
so a human can label a corrective action with enough context to make a real
decision.  It never connects to the game and never sends controller input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v4.dagger_corrective import ACTION_NAMES, _load_queue


def _script_json(value: Any) -> str:
    """Serialize queue data safely for an inline browser script."""

    return (
        json.dumps(value, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _review_html(queue: list[dict[str, Any]]) -> str:
    payload = _script_json(queue)
    action_payload = _script_json(list(ACTION_NAMES))
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DAgger corrective-state review</title>
<style>
:root { color-scheme: dark; --bg:#111827; --panel:#1f2937; --muted:#9ca3af; --text:#f3f4f6; --line:#374151; --accent:#60a5fa; --danger:#f87171; --warn:#fbbf24; --good:#34d399; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; }
header { padding:16px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
h1 { margin:0; font-size:20px; }
header p { margin:0; color:var(--muted); }
button, input { font:inherit; }
button { border:1px solid var(--line); border-radius:6px; color:var(--text); background:#293548; padding:7px 10px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.primary { background:#1d4ed8; border-color:#3b82f6; }
button.selected { background:#047857; border-color:#34d399; }
button.skip { background:#7f1d1d; border-color:#ef4444; }
button:disabled { opacity:.45; cursor:not-allowed; }
main { display:grid; grid-template-columns:minmax(520px, 1.45fr) minmax(360px, .85fr); gap:16px; padding:16px; max-width:1600px; margin:0 auto; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
.map-panel { min-width:0; }
canvas { width:100%; height:auto; display:block; background:#0b1220; border:1px solid #4b5563; border-radius:6px; }
.legend { display:flex; gap:13px; flex-wrap:wrap; color:var(--muted); margin-top:8px; font-size:12px; }
.legend span { display:inline-flex; align-items:center; gap:5px; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot.player { background:#60a5fa; }.dot.enemy { background:#f87171; }.dot.charge { background:#fbbf24; }.dot.projectile { background:#fb7185; }.dot.pickup { background:#34d399; }
.toolbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
.toolbar strong { margin-right:auto; }
.progress { color:var(--muted); }
.field { display:grid; grid-template-columns:145px 1fr; gap:5px 10px; margin:8px 0; }
.field dt { color:var(--muted); }.field dd { margin:0; word-break:break-word; }
.value-danger { color:var(--danger); }.value-warn { color:var(--warn); }.value-good { color:var(--good); }
.actions { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:7px; margin:10px 0; }
.actions button { min-height:38px; }
.label-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:10px 0; }
.label-row input { width:120px; background:#111827; color:var(--text); border:1px solid var(--line); border-radius:5px; padding:7px; }
.reasons { display:flex; gap:5px; flex-wrap:wrap; }
.reason { color:#bfdbfe; background:#172554; border:1px solid #1e40af; border-radius:4px; padding:3px 6px; font-size:12px; }
pre { max-height:240px; overflow:auto; white-space:pre-wrap; word-break:break-word; color:#d1d5db; background:#111827; padding:8px; border-radius:5px; }
.note { color:var(--muted); font-size:12px; }
@media (max-width: 980px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>DAgger corrective-state review</h1>
  <p>Offline tactical reconstruction from the exact captured state. No game control is active.</p>
</header>
<main>
  <section class="panel map-panel">
    <div class="toolbar">
      <strong id="title">State</strong>
      <span class="progress" id="progress"></span>
      <button id="previous">Previous</button>
      <button id="next">Next</button>
    </div>
    <canvas id="map" width="1000" height="700" aria-label="Recorded battlefield tactical map"></canvas>
    <div class="legend">
      <span><i class="dot player"></i>player</span>
      <span><i class="dot enemy"></i>enemy</span>
      <span><i class="dot charge"></i>charging enemy</span>
      <span><i class="dot projectile"></i>hostile projectile</span>
      <span><i class="dot pickup"></i>pickup</span>
    </div>
    <p class="note">This is a tactical map, not a pixel screenshot. Unknown-size objects are omitted from the map but remain in the recorded state details. Use replay/game review for any case where the geometry is ambiguous.</p>
  </section>
  <section class="panel">
    <div class="toolbar">
      <strong>Decision context</strong>
      <span class="progress" id="labelProgress"></span>
    </div>
    <dl class="field">
      <dt>Queue ID</dt><dd id="queueId"></dd>
      <dt>Split</dt><dd id="split"></dd>
      <dt>Episode / tick</dt><dd id="episodeTick"></dd>
      <dt>Timestamp</dt><dd id="timestamp"></dd>
      <dt>Build / wave</dt><dd id="buildWave"></dd>
      <dt>Player HP</dt><dd id="health"></dd>
      <dt>Enemies / projectiles</dt><dd id="density"></dd>
      <dt>Current action</dt><dd id="current"></dd>
      <dt>Handcrafted recommendation</dt><dd id="handcrafted"></dd>
      <dt>Learned proposal</dt><dd id="learned"></dd>
      <dt>Model confidence</dt><dd id="confidence"></dd>
      <dt>Risk comparison</dt><dd id="risk"></dd>
      <dt>Safety</dt><dd id="safety"></dd>
    </dl>
    <div class="reasons" id="reasons"></div>
    <h3>Human corrective action</h3>
    <div class="actions" id="actions"></div>
    <div class="label-row">
      <label for="hold">Optional hold (ms)</label>
      <input id="hold" type="number" min="0" step="1" placeholder="unknown">
      <button class="skip" id="skip">Skip this state</button>
    </div>
    <p id="selectedLabel" class="note">No human label recorded for this state.</p>
    <div class="toolbar">
      <button class="primary" id="download">Download labels JSONL</button>
      <label>Load prior labels <input id="load" type="file" accept=".jsonl,.json,.txt"></label>
    </div>
    <p class="note">Choose the action you would actually take at the recorded pre-action moment. Do not copy the model merely because it is confident. Skipped rows are not exported as training labels.</p>
    <details><summary>Raw captured state</summary><pre id="raw"></pre></details>
  </section>
</main>
<script>
const QUEUE = __QUEUE_JSON__;
const ACTIONS = __ACTION_JSON__;
let index = 0;
const labels = new Map();
const skipped = new Set();

const $ = (id) => document.getElementById(id);
const finite = (value, fallback = 0) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
const text = (id, value) => { $(id).textContent = value == null ? "unknown" : String(value); };
const actionName = (value) => {
  const n = Number(value);
  return Number.isInteger(n) && n >= 0 && n < ACTIONS.length ? `${n} ${ACTIONS[n]}` : "unknown";
};
const stateOf = (item) => item.state || {};
const contextOf = (item) => (item.shadow_record && item.shadow_record.context) || {};
const posOf = (entity) => {
  const p = entity && entity.position;
  if (!p) return null;
  const x = Number(p.x), y = Number(p.y);
  return Number.isFinite(x) && Number.isFinite(y) ? {x, y} : null;
};
const validWorldPos = (p, width, height) => p && p.x >= 0 && p.y >= 0 && p.x <= width && p.y <= height;

function transformFor(state, canvas) {
  const arena = state.arena || {};
  const width = Math.max(1, finite(arena.width, 2048));
  const height = Math.max(1, finite(arena.height, 1536));
  const margin = 34;
  const scale = Math.min((canvas.width - margin * 2) / width, (canvas.height - margin * 2) / height);
  const ox = (canvas.width - width * scale) / 2;
  const oy = (canvas.height - height * scale) / 2;
  return {width, height, scale, ox, oy, point: (p) => ({x: ox + p.x * scale, y: oy + p.y * scale})};
}

function drawArrow(ctx, a, b, color) {
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - 10 * Math.cos(angle - Math.PI / 6), b.y - 10 * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(b.x - 10 * Math.cos(angle + Math.PI / 6), b.y - 10 * Math.sin(angle + Math.PI / 6));
  ctx.closePath(); ctx.fill();
}

function drawMap(item) {
  const canvas = $("map"), ctx = canvas.getContext("2d"), state = stateOf(item), t = transformFor(state, canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0b1220"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#334155"; ctx.lineWidth = 1; ctx.strokeRect(t.ox, t.oy, t.width * t.scale, t.height * t.scale);
  ctx.strokeStyle = "#1e293b";
  for (let x = 0; x <= t.width; x += 256) { const px = t.ox + x * t.scale; ctx.beginPath(); ctx.moveTo(px, t.oy); ctx.lineTo(px, t.oy + t.height * t.scale); ctx.stroke(); }
  for (let y = 0; y <= t.height; y += 256) { const py = t.oy + y * t.scale; ctx.beginPath(); ctx.moveTo(t.ox, py); ctx.lineTo(t.ox + t.width * t.scale, py); ctx.stroke(); }
  const drawEntity = (entity, color, radius, label) => {
    const p = posOf(entity); if (!validWorldPos(p, t.width, t.height)) return;
    const q = t.point(p); const r = Math.max(3, Math.min(34, finite(radius, 12) * t.scale));
    ctx.fillStyle = color; ctx.globalAlpha = .84; ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
    if (label) { ctx.fillStyle = "#f8fafc"; ctx.font = "11px system-ui"; ctx.textAlign = "center"; ctx.fillText(label, q.x, q.y + 4); }
  };
  for (const pickup of (state.pickups || [])) drawEntity(pickup, "#34d399", finite(pickup.radius, 10), "P");
  for (const projectile of (state.projectiles || [])) {
    if (!projectile.hostile || projectile.size_known === false) continue;
    drawEntity(projectile, "#fb7185", finite(projectile.radius, 6), "!");
  }
  for (const indicator of (state.attack_indicators || [])) {
    if (indicator.size_known === false) continue;
    const p = posOf(indicator); if (!validWorldPos(p, t.width, t.height)) continue;
    const q = t.point(p); const r = Math.max(5, finite(indicator.radius, 12) * t.scale);
    ctx.strokeStyle = "#f43f5e"; ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
  }
  for (const enemy of (state.enemies || [])) {
    const color = enemy.is_charging ? "#fbbf24" : (enemy.is_boss || enemy.is_elite ? "#dc2626" : "#f87171");
    drawEntity(enemy, color, finite(enemy.radius, 14), enemy.is_boss ? "B" : (enemy.is_elite ? "E" : ""));
  }
  const player = state.player || {}, playerPos = posOf(player);
  if (validWorldPos(playerPos, t.width, t.height)) {
    const q = t.point(playerPos); const r = Math.max(7, finite(player.radius, 21) * t.scale);
    ctx.fillStyle = "#60a5fa"; ctx.strokeStyle = "#dbeafe"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    const v = player.velocity || {}; const vx = finite(v.x), vy = finite(v.y); const length = Math.hypot(vx, vy);
    if (length > 1) { const factor = Math.min(80, 20 + length / 10); drawArrow(ctx, q, {x: q.x + vx / length * factor, y: q.y + vy / length * factor}, "#bfdbfe"); }
  }
  ctx.fillStyle = "#cbd5e1"; ctx.font = "12px system-ui"; ctx.textAlign = "left";
  ctx.fillText(`arena ${Math.round(t.width)} × ${Math.round(t.height)}`, 12, 20);
}

function riskText(item) {
  const shadow = item.shadow_record || {}, h = shadow.human_risk || {}, b = shadow.handcrafted_risk || {};
  const delta = shadow.human_minus_handcrafted_risk;
  const parts = [];
  if (Number.isFinite(Number(b.total))) parts.push(`baseline ${finite(b.total).toFixed(3)}`);
  if (Number.isFinite(Number(h.total))) parts.push(`learned ${finite(h.total).toFixed(3)}`);
  if (Number.isFinite(Number(delta))) parts.push(`Δ ${finite(delta) >= 0 ? "+" : ""}${finite(delta).toFixed(3)}`);
  return parts.join(" · ") || "not available";
}

function refreshProgress() {
  const labeled = labels.size, total = QUEUE.length;
  const train = [...labels.keys()].filter((id) => QUEUE.find((item) => item.queue_id === id)?.split === "train").length;
  const holdout = labeled - train;
  text("progress", `${index + 1}/${total}`);
  text("labelProgress", `labeled ${labeled}/${total} · train ${train} · holdout ${holdout}`);
}

function render() {
  if (!QUEUE.length) return;
  const item = QUEUE[index], state = stateOf(item), context = contextOf(item), player = state.player || {};
  text("title", `State ${index + 1}: ${item.queue_id}`);
  text("queueId", item.queue_id); text("split", item.split);
  text("episodeTick", `${item.source_episode} / ${item.source_tick}`);
  text("timestamp", `${finite(item.source_timestamp_ms).toFixed(1)} ms`);
  text("buildWave", `${item.build || context.build || "unknown"} / wave ${item.wave || context.wave || "?"}`);
  const hp = finite(player.health, finite(context.health, NaN)), maxHp = finite(player.max_health, finite(context.max_health, NaN));
  text("health", Number.isFinite(hp) && Number.isFinite(maxHp) ? `${hp.toFixed(1)} / ${maxHp.toFixed(1)} (${(100 * hp / Math.max(1, maxHp)).toFixed(1)}%)` : "unknown");
  text("density", `${(state.enemies || []).length} enemies / ${(state.projectiles || []).filter((p) => p.hostile).length} hostile projectiles / ${(state.attack_indicators || []).length} indicators`);
  text("current", actionName(item.current_action)); text("handcrafted", actionName(item.handcrafted_recommendation));
  text("learned", actionName(item.learned_proposal)); text("confidence", item.model_confidence == null ? "unknown" : finite(item.model_confidence).toFixed(3));
  text("risk", riskText(item));
  const safety = item.safety || (item.shadow_record || {}).safety || {};
  text("safety", safety.human_would_override == null ? "not available" : (safety.human_would_override ? `would override → ${actionName(safety.human_would_apply_action)}` : "would not override"));
  const reasons = $("reasons"); reasons.replaceChildren();
  for (const reason of (item.selection_reasons || [])) { const span = document.createElement("span"); span.className = "reason"; span.textContent = reason; reasons.appendChild(span); }
  const actionBox = $("actions"); actionBox.replaceChildren();
  const currentLabel = labels.get(item.queue_id);
  for (let action = 0; action < ACTIONS.length; action++) {
    const button = document.createElement("button"); button.textContent = `${action} ${ACTIONS[action]}`;
    if (currentLabel && Number(currentLabel.action) === action) button.classList.add("selected");
    button.onclick = () => labelAction(action); actionBox.appendChild(button);
  }
  $("hold").value = currentLabel && currentLabel.hold_duration_ms != null ? currentLabel.hold_duration_ms : "";
  text("selectedLabel", currentLabel ? `Human label: ${actionName(currentLabel.action)}${currentLabel.hold_duration_ms == null ? "" : ` for ${currentLabel.hold_duration_ms} ms`}.` : (skipped.has(item.queue_id) ? "Marked skipped; not exported." : "No human label recorded for this state."));
  $("raw").textContent = JSON.stringify(state, null, 2);
  $("previous").disabled = index <= 0; $("next").disabled = index >= QUEUE.length - 1;
  drawMap(item); refreshProgress();
}

function labelAction(action) {
  const item = QUEUE[index], holdText = $("hold").value.trim();
  const hold = holdText === "" ? null : Number(holdText);
  if (hold != null && (!Number.isFinite(hold) || hold < 0)) { alert("Hold duration must be a non-negative number or blank."); return; }
  labels.set(item.queue_id, {queue_id: item.queue_id, human_corrective_action: ACTIONS[action], hold_duration_ms: hold});
  skipped.delete(item.queue_id); render();
}
function skipCurrent() { const item = QUEUE[index]; labels.delete(item.queue_id); skipped.add(item.queue_id); render(); }
function downloadLabels() {
  const lines = [...labels.values()].map((row) => JSON.stringify(row));
  const blob = new Blob([lines.join("\n") + (lines.length ? "\n" : "")], {type:"application/jsonl"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "dagger_corrective_labels.jsonl"; a.click(); URL.revokeObjectURL(a.href);
}
function loadLabels(file) {
  const reader = new FileReader(); reader.onload = () => {
    const lines = String(reader.result || "").split(/\r?\n/); let errors = 0;
    for (const line of lines) { if (!line.trim()) continue; try { const row = JSON.parse(line); if (row.queue_id && row.human_corrective_action != null) labels.set(row.queue_id, row); else errors++; } catch (_) { errors++; } }
    render(); if (errors) alert(`${errors} label lines could not be loaded.`);
  }; reader.readAsText(file);
}
$("previous").onclick = () => { if (index > 0) { index--; render(); } };
$("next").onclick = () => { if (index + 1 < QUEUE.length) { index++; render(); } };
$("skip").onclick = skipCurrent; $("download").onclick = downloadLabels;
$("load").onchange = (event) => { const file = event.target.files && event.target.files[0]; if (file) loadLabels(file); };
document.addEventListener("keydown", (event) => {
  if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
  if (/^[0-8]$/.test(event.key)) labelAction(Number(event.key));
  else if (event.key.toLowerCase() === "s") skipCurrent();
  else if (event.key === "ArrowLeft" && index > 0) { index--; render(); }
  else if (event.key === "ArrowRight" && index + 1 < QUEUE.length) { index++; render(); }
});
render();
</script>
</body>
</html>
'''
    return template.replace("__QUEUE_JSON__", payload).replace("__ACTION_JSON__", action_payload)


def render_review_html(queue_path: Path, output_path: Path) -> dict[str, Any]:
    """Write a self-contained tactical review page for a corrective queue."""

    queue = _load_queue(queue_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_review_html(queue), encoding="utf-8")
    return {
        "queue": str(queue_path),
        "output": str(output_path),
        "rows": len(queue),
        "visualization": "offline tactical map reconstructed from captured coordinates; not a pixel screenshot",
        "labels_created": 0,
        "synthetic_labels": 0,
        "production_control_changed": False,
    }
