/* ==========================================================================
   core.js - shared state, API client, formatting helpers, chart defaults
   ========================================================================== */
'use strict';

const APP = {
  meta: null,
  summary: null,
  points: null,
  charts: {},
  state: {
    dose: null, energy: null, temp: null, time: null,
    lastPredict: null, lastOptimize: null, lastSweep: null,
  },
};

/* ------------------------------------------------------------------ api */
const API = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let detail;
      try { detail = (await r.json()).detail; } catch (_) { detail = await r.text(); }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return r.json();
  },
};

/* --------------------------------------------------------------- format */
const COLORS = {
  blue: '#58a6ff', cyan: '#39d353', green: '#3fb950', yellow: '#d29922',
  orange: '#db6d28', red: '#f85149', purple: '#a371f7', pink: '#db61a2',
  grid: 'rgba(110,118,129,0.18)', text: '#8b949e',
};
const SERIES = [COLORS.blue, COLORS.purple, COLORS.cyan, COLORS.yellow, COLORS.orange, COLORS.pink];

function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(Math.max(0, digits - 2));
  return Number(v.toPrecision(digits)).toString();
}
function fmtSci(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return Number(v).toExponential(digits - 1).replace('e+', 'e');
}
function fmtFixed(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return Number(v).toFixed(d);
}
function fmtInt(v) {
  if (v === null || v === undefined) return '–';
  return Number(v).toLocaleString();
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function paramText(key, value) {
  if (key === 'dose_cm2') return fmtSci(value, 3) + ' cm⁻²';
  if (key === 'energy_keV') return fmt(value, 4) + ' keV';
  if (key === 'anneal_temp_C') return fmt(value, 5) + ' °C';
  if (key === 'anneal_time_sec') return fmt(value, 4) + ' sec';
  return fmt(value);
}
const PARAM_LABEL = {
  dose_cm2: 'Implant Dose', energy_keV: 'Implant Energy',
  anneal_temp_C: 'Anneal Temperature', anneal_time_sec: 'Anneal Time',
};
const PARAM_LABEL_KO = {
  dose_cm2: '이온주입 도즈', energy_keV: '이온주입 에너지',
  anneal_temp_C: '열처리 온도', anneal_time_sec: '열처리 시간',
};
const PARAM_UNIT = { dose_cm2: 'cm⁻²', energy_keV: 'keV', anneal_temp_C: '°C', anneal_time_sec: 'sec' };
const TARGET_LABEL = {
  xj_implant_um: 'Xj Implant', xj_final_um: 'Xj Final',
  delta_xj_um: 'ΔXj', rsh_final_ohm_sq: 'Rsh Final',
};
const TARGET_UNIT = {
  xj_implant_um: 'um', xj_final_um: 'um', delta_xj_um: 'um', rsh_final_ohm_sq: 'ohm/sq',
};

/* --------------------------------------------------------------- charts */
if (window.Chart) {
  Chart.defaults.color = COLORS.text;
  Chart.defaults.font.family = "'Inter','Malgun Gothic',sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = '#161b22';
  Chart.defaults.plugins.tooltip.borderColor = '#30363d';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = '#e6edf3';
  Chart.defaults.plugins.tooltip.bodyColor = '#8b949e';
  Chart.defaults.plugins.tooltip.padding = 9;
  Chart.defaults.maintainAspectRatio = false;
}

function axis(title, opts = {}) {
  return Object.assign({
    title: { display: !!title, text: title, color: COLORS.text, font: { size: 10.5 } },
    grid: { color: COLORS.grid },
    ticks: { color: COLORS.text, font: { size: 10 } },
  }, opts);
}

function drawChart(id, config) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (APP.charts[id]) { APP.charts[id].destroy(); }
  APP.charts[id] = new Chart(el.getContext('2d'), config);
  return APP.charts[id];
}

function hexA(hex, alpha) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/* map value -> color on a blue->cyan->yellow->red ramp */
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.0, [40, 90, 200]], [0.35, [56, 166, 255]],
    [0.6, [57, 211, 83]], [0.8, [210, 153, 34]], [1.0, [248, 81, 73]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      return `rgb(${Math.round(c0[0] + f * (c1[0] - c0[0]))},${Math.round(c0[1] + f * (c1[1] - c0[1]))},${Math.round(c0[2] + f * (c1[2] - c0[2]))})`;
    }
  }
  return COLORS.blue;
}
function divergent(v) {
  // -1..1 -> red .. neutral .. blue
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const a = 0.12 + 0.78 * t;
    return `rgba(88,166,255,${a.toFixed(3)})`;
  }
  const a = 0.12 + 0.78 * (-t);
  return `rgba(248,81,73,${a.toFixed(3)})`;
}

/* -------------------------------------------------------------- widgets */
function bannerHtml(kind, icon, title, text) {
  return `<div class="banner ${kind}"><span class="banner-icon">${icon}</span><div><b>${esc(title)}</b> ${esc(text)}</div></div>`;
}

function barRows(items, valueFn, labelFn, colorClass = 'pos') {
  const max = Math.max(...items.map(i => Math.abs(valueFn(i)))) || 1;
  return items.map(i => {
    const v = valueFn(i);
    const w = (Math.abs(v) / max) * 100;
    const cls = colorClass === 'auto' ? (v >= 0 ? 'pos' : 'neg') : colorClass;
    return `<div class="bar-row">
      <div class="bar-head"><span class="bar-name">${esc(labelFn(i))}</span><span class="bar-num">${esc(i.numText)}</span></div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${w.toFixed(1)}%"></div></div>
    </div>`;
  }).join('');
}

function insightCards(list) {
  if (!list || !list.length) return '<div class="empty-state">–</div>';
  const cls = { 'DATA OBSERVATION': 'level-data', 'MODEL INTERPRETATION': 'level-model', 'ENGINEERING VERIFICATION REQUIRED': 'level-verify' };
  return list.map(c => `
    <div class="insight-card ${cls[c.level] || ''}">
      <span class="insight-level">${esc(c.level)}</span>
      <div class="insight-title">${esc(c.title_ko)}</div>
      <div class="insight-text">${esc(c.text_ko)}<span class="en">${esc(c.text_en)}</span></div>
    </div>`).join('');
}

function metricRows(rows) {
  return `<div class="metric-table">${rows.map(([k, v]) =>
    `<div class="metric-row"><span class="k">${esc(k)}</span><span class="v">${v}</span></div>`).join('')}</div>`;
}

function setStatus(kind, text) {
  const el = document.getElementById('modelStatus');
  el.className = 'status-badge' + (kind === 'ok' ? '' : ' ' + kind);
  document.getElementById('modelStatusText').textContent = text;
}

/* ----------------------------------------------------------- log slider */
/* dose slider works in log10 space so the 5e14..2.5e15 span is uniform */
function logSliderToValue(t, lo, hi) {
  return Math.pow(10, Math.log10(lo) + t * (Math.log10(hi) - Math.log10(lo)));
}
function valueToLogSlider(v, lo, hi) {
  return (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
}
function nearest(arr, v) {
  return arr.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));
}
