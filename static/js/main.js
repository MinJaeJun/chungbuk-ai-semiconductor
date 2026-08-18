/* ==========================================================================
   main.js - bootstrap, tab routing, KPI strip
   ========================================================================== */
'use strict';

const TAB_INIT = {
  explorer: false, predict: false, whatif: false,
  optimizer: false, xai: false, robust: false, validation: false,
  rigor: false,
};

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  lazyInit(name);
  // charts sized while hidden need a nudge
  setTimeout(() => Object.values(APP.charts).forEach(c => { try { c.resize(); } catch (_) {} }), 30);
}

async function lazyInit(name) {
  if (TAB_INIT[name]) return;
  TAB_INIT[name] = true;
  try {
    if (name === 'explorer') Explorer.init();
    else if (name === 'predict') Predict.init();
    else if (name === 'whatif') { WhatIf.init(); await WhatIf.run(); }
    else if (name === 'optimizer') { Optimizer.init(); await Optimizer.run(); }
    else if (name === 'xai') await Xai.init();
    else if (name === 'robust') await Robust.init();
    else if (name === 'validation') await Validation.init();
    else if (name === 'rigor') await Rigor.init();
  } catch (e) {
    TAB_INIT[name] = false;
    const panel = document.getElementById(`tab-${name}`);
    panel.insertAdjacentHTML('afterbegin',
      `<div class="banner error"><span class="banner-icon">⚠</span><div><b>${esc(name)} 탭 로딩 실패.</b> ${esc(e.message)}</div></div>`);
    console.error(name, e);
  }
}

function renderKpi() {
  const m = APP.meta;
  const v = APP.summary.validation;
  const strip = document.getElementById('kpiStrip');
  if (!m.model_trained) {
    strip.innerHTML = `<div class="kpi-card yellow" style="grid-column:1/-1">
      <div class="kpi-label">MODEL NOT TRAINED</div>
      <div class="kpi-value sm">python train_model.py 를 먼저 실행하세요</div></div>`;
    return;
  }
  const xj = m.model.metrics.xj_final_um;
  const rsh = m.model.metrics.rsh_final_ohm_sq;
  // Model selection uses two different documented criteria, so the label must
  // reflect the one actually applied to that target.
  const selLabel = (t) => {
    const s = (m.model.selection_metrics || {})[t] || '';
    return s.startsWith('derived') ? 'selected by derived ΔXj RMSE (pair)' : 'selected by group CV R²';
  };
  strip.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">TCAD DATA</div>
      <div class="kpi-value">${fmtInt(v.n_rows)}<span class="kpi-unit">Runs</span></div>
      <div class="kpi-sub">${esc(v.doe.structure)} full factorial</div>
    </div>
    <div class="kpi-card cyan">
      <div class="kpi-label">PROCESS VARIABLES</div>
      <div class="kpi-value">${v.n_inputs}<span class="kpi-unit">Parameters</span></div>
      <div class="kpi-sub">dose · energy · temp · time</div>
    </div>
    <div class="kpi-card purple">
      <div class="kpi-label">AI MODEL · Xj Final</div>
      <div class="kpi-value sm">${esc(m.model.best_models.xj_final_um)}</div>
      <div class="kpi-sub">${esc(selLabel('xj_final_um'))}</div>
    </div>
    <div class="kpi-card purple">
      <div class="kpi-label">AI MODEL · Rsh</div>
      <div class="kpi-value sm">${esc(m.model.best_models.rsh_final_ohm_sq)}</div>
      <div class="kpi-sub">${esc(selLabel('rsh_final_ohm_sq'))}</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">PREDICTION · Xj Final R²</div>
      <div class="kpi-value">${xj.test.r2.toFixed(5)}</div>
      <div class="kpi-sub">MAE ${fmt(xj.test.mae, 3)} um · hold-out test</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">PREDICTION · Rsh R²</div>
      <div class="kpi-value">${rsh.test.r2.toFixed(5)}</div>
      <div class="kpi-sub">MAE ${fmt(rsh.test.mae, 3)} ohm/sq · hold-out test</div>
    </div>`;
}

function exportJson() {
  const payload = {
    exported_at: new Date().toISOString(),
    app: APP.meta.app,
    model: APP.meta.model || null,
    operating_point: {
      dose_cm2: APP.state.dose, energy_keV: APP.state.energy,
      anneal_temp_C: APP.state.temp, anneal_time_sec: APP.state.time,
    },
    last_prediction: APP.state.lastPredict,
    last_optimization: APP.state.lastOptimize,
    disclaimer_en: APP.meta.disclaimer_en,
    disclaimer_ko: APP.meta.disclaimer_ko,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `ai_process_optimizer_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

(async function boot() {
  document.querySelectorAll('.tab').forEach(t =>
    t.addEventListener('click', () => switchTab(t.dataset.tab)));
  document.getElementById('btnExport').addEventListener('click', exportJson);

  try {
    setStatus('warn', 'loading…');
    const [meta, summary, points] = await Promise.all([
      API.get('/api/meta'), API.get('/api/dataset/summary'), API.get('/api/dataset/points'),
    ]);
    APP.meta = meta; APP.summary = summary; APP.points = points;

    // Default operating point (a real DOE grid point) so any tab can be opened
    // first without depending on the Prediction tab having been initialised.
    const lv = meta.dataset.doe_levels;
    APP.state.dose = nearest(lv.dose_cm2, 1.5e15);
    APP.state.energy = nearest(lv.energy_keV, 20);
    APP.state.temp = nearest(lv.anneal_temp_C, 1000);
    APP.state.time = nearest(lv.anneal_time_sec, 25);

    if (!meta.model_trained) {
      setStatus('err', 'model not trained');
    } else {
      setStatus('ok', `${meta.model.best_models.xj_final_um.split(' ')[0]} / ${meta.model.best_models.rsh_final_ohm_sq.split(' ')[0]} · trained`);
    }
    renderKpi();
    lazyInit('explorer');
  } catch (e) {
    setStatus('err', 'backend error');
    document.querySelector('main').insertAdjacentHTML('afterbegin',
      `<div style="padding:24px"><div class="banner error"><span class="banner-icon">⚠</span>
       <div><b>백엔드 연결 실패.</b> ${esc(e.message)}<br>
       <span class="mono small">uvicorn app:app --port 8000 이 실행 중인지, python train_model.py 로 모델이 학습되었는지 확인하세요.</span></div></div></div>`);
    console.error(e);
  }
})();
