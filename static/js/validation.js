/* ==========================================================================
   validation.js - Model Validation tab (judge-facing evidence)
   ========================================================================== */
'use strict';

const Validation = {
  async init() {
    const v = await API.get('/api/validation');
    APP.validation = v;

    document.getElementById('valDisclaimer').innerHTML = `
      <span class="banner-icon">⚠</span>
      <div><b>${esc(v.disclaimer_en)}</b><br>${esc(v.disclaimer_ko)}</div>`;

    const cov = v.dataset_coverage;
    document.getElementById('valCovBadge').textContent = `${fmtInt(cov.total_runs)} TCAD runs`;
    const t0 = v.targets.xj_final_um;
    document.getElementById('valCoverage').innerHTML = `
      <div class="grid-4 mb12">
        ${[
          ['TOTAL TCAD RUNS', fmtInt(cov.total_runs), 'runs', cov.doe.structure],
          ['TRAIN / TEST', `${t0.n_train} / ${t0.n_test}`, 'rows', `random split, seed 42 (80/20)`],
          ['CROSS VALIDATION', `${t0.cv_folds}-fold`, '', `+ GroupKFold on ${t0.n_groups} implant conditions`],
          ['MODEL SELECTION', 'group CV R²', '', esc(t0.selection_metric)],
        ].map(([l, val, u, sub], i) => `
          <div class="kpi-card ${['', 'cyan', 'purple', 'yellow'][i]}">
            <div class="kpi-label">${l}</div>
            <div class="kpi-value ${String(val).length > 9 ? 'sm' : ''}">${val}<span class="kpi-unit">${u}</span></div>
            <div class="kpi-sub">${sub}</div>
          </div>`).join('')}
      </div>
      <div class="grid-2">
        <div>
          <div class="param-section-title">Input coverage (training envelope)</div>
          ${metricRows(Object.entries(cov.input_bounds).map(([k, b]) =>
            [PARAM_LABEL[k], `${k === 'dose_cm2' ? fmtSci(b.min, 3) : fmt(b.min, 5)} ~ ${k === 'dose_cm2' ? fmtSci(b.max, 3) : fmt(b.max, 5)} ${PARAM_UNIT[k]}`]))}
        </div>
        <div>
          <div class="param-section-title">Output coverage</div>
          ${metricRows(Object.entries(cov.target_bounds).map(([k, b]) =>
            [TARGET_LABEL[k] || k, `${fmt(b.min, 4)} ~ ${fmt(b.max, 4)} ${TARGET_UNIT[k] || ''}`]))}
        </div>
      </div>
      <div class="banner info mt12" style="margin-bottom:0">
        <span class="banner-icon">🔒</span>
        <div><b>Data leakage policy.</b> ${esc(cov.leakage_policy.note)}<br>
        <span class="mono small">features = [${cov.leakage_policy.features_used.join(', ')}] · excluded = [${cov.leakage_policy.excluded_from_features.join(', ')}]</span></div>
      </div>`;

    const order = ['xj_final_um', 'rsh_final_ohm_sq', 'xj_implant_um'];
    document.getElementById('valTargets').innerHTML = order.map(t => {
      const b = v.targets[t];
      const m = b.metrics;
      return `
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">🎯 ${esc(b.label)} · ${esc(b.label_ko)} [${esc(b.unit)}]</span>
          <span class="panel-badge purple">BEST: ${esc(b.best_model)}</span>
        </div>
        <div class="panel-content">
          <div class="grid-3 mb12">
            ${[
              ['GROUP CV (unseen condition)', m.group_cv, 'yellow'],
              ['RANDOM CV (5-fold)', m.cv, 'cyan'],
              ['HOLD-OUT TEST (20%)', m.test, ''],
            ].map(([lab, mm, c]) => `
              <div class="kpi-card ${c}">
                <div class="kpi-label">${lab}</div>
                <div class="kpi-value">R² ${mm.r2.toFixed(6)}</div>
                <div class="kpi-sub">MAE ${fmt(mm.mae, 4)} · RMSE ${fmt(mm.rmse, 4)} ${esc(b.unit)}</div>
              </div>`).join('')}
          </div>
          <div class="grid-2">
            <div class="chart-box"><canvas id="valScatter_${t}"></canvas></div>
            <div class="chart-box"><canvas id="valResid_${t}"></canvas></div>
          </div>
          <div class="param-section-title mt16">Model comparison leaderboard (실제 측정값)</div>
          <div class="table-wrap">
            <table class="grid-table">
              <thead><tr>
                <th>#</th><th>Model</th>
                <th>group CV R²</th><th>group CV MAE</th><th>group CV RMSE</th>
                <th>rand CV R²</th>
                <th>test R²</th><th>test MAE</th><th>test RMSE</th><th>test MAPE %</th><th>fit [s]</th>
              </tr></thead>
              <tbody>${b.leaderboard.map((r, i) => `
                <tr>
                  <td><span class="rank-pill ${i === 0 ? 'top' : ''}">${i + 1}</span></td>
                  <td class="name">${esc(r.model)}${i === 0 ? ' <span class="pill ok">SELECTED</span>' : ''}</td>
                  <td>${r.group_cv_r2.toFixed(6)}</td>
                  <td>${fmt(r.group_cv_mae, 4)}</td>
                  <td>${fmt(r.group_cv_rmse, 4)}</td>
                  <td class="muted">${r.cv_r2.toFixed(6)}</td>
                  <td>${r.test_r2.toFixed(6)}</td>
                  <td>${fmt(r.test_mae, 4)}</td>
                  <td>${fmt(r.test_rmse, 4)}</td>
                  <td class="muted">${fmt(r.test_mape_pct, 3)}</td>
                  <td class="muted">${r.fit_seconds}</td>
                </tr>`).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>`;
    }).join('');

    order.forEach(t => Validation.plots(t, v.targets[t]));

    const env = v.environment;
    document.getElementById('valFooter').innerHTML = `
      <b>${esc(v.disclaimer_en)}</b><br>
      ${esc(v.disclaimer_ko)}<br><br>
      <span class="mono small">trained_at ${esc(v.generated_at)} · python ${esc(env.python)} · scikit-learn ${esc(env.scikit_learn)} ·
      numpy ${esc(env.numpy)} · pandas ${esc(env.pandas)} · optional libs ${esc(JSON.stringify(env.optional_libraries))}</span><br>
      <span class="small">이 페이지의 모든 R² / MAE / RMSE 값은 <span class="mono">train_model.py</span> 실행 시 실제 데이터에서 계산되어
      <span class="mono">outputs/training_report.json</span>에 저장된 값을 그대로 표시한 것입니다. 하드코딩된 성능 수치는 없습니다.</span>`;
  },

  plots(t, b) {
    const unit = b.unit;
    const yt = b.validation.y_true, yp = b.validation.y_pred;
    const gt = b.group_validation.y_true, gp = b.group_validation.y_pred;
    const lo = Math.min(...yt, ...yp), hi = Math.max(...yt, ...yp);

    drawChart(`valScatter_${t}`, {
      type: 'scatter',
      data: {
        datasets: [
          { label: `Group CV (unseen condition, n=${gt.length})`, data: gt.map((v, i) => ({ x: v, y: gp[i] })), pointRadius: 2, backgroundColor: 'rgba(210,153,34,.42)', borderWidth: 0 },
          { label: `Hold-out test (n=${yt.length})`, data: yt.map((v, i) => ({ x: v, y: yp[i] })), pointRadius: 3.2, backgroundColor: 'rgba(88,166,255,.8)', borderWidth: 0 },
          { label: 'y = x (ideal)', data: [{ x: lo, y: lo }, { x: hi, y: hi }], showLine: true, pointRadius: 0, borderColor: 'rgba(248,81,73,.85)', borderDash: [6, 4], borderWidth: 1.6 },
        ],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: `Actual vs Predicted · ${b.label}`, color: '#8b949e', font: { size: 11 } },
          tooltip: { callbacks: { label: c => `actual ${fmt(c.parsed.x, 5)} → pred ${fmt(c.parsed.y, 5)} ${unit}` } },
        },
        scales: { x: axis(`Actual (TCAD) [${unit}]`), y: axis(`Predicted (AI) [${unit}]`) },
      },
    });

    drawChart(`valResid_${t}`, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Group CV residual', data: gt.map((v, i) => ({ x: v, y: gp[i] - v })), pointRadius: 2, backgroundColor: 'rgba(210,153,34,.42)', borderWidth: 0 },
          { label: 'Test residual', data: yt.map((v, i) => ({ x: v, y: yp[i] - v })), pointRadius: 3.2, backgroundColor: 'rgba(163,113,247,.8)', borderWidth: 0 },
          { label: 'zero', data: [{ x: lo, y: 0 }, { x: hi, y: 0 }], showLine: true, pointRadius: 0, borderColor: 'rgba(110,118,129,.7)', borderWidth: 1.4 },
        ],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: `Residual (pred − actual) · ${b.label}`, color: '#8b949e', font: { size: 11 } },
          tooltip: { callbacks: { label: c => `actual ${fmt(c.parsed.x, 5)} → residual ${fmt(c.parsed.y, 4)} ${unit}` } },
        },
        scales: { x: axis(`Actual (TCAD) [${unit}]`), y: axis(`Residual [${unit}]`) },
      },
    });
  },
};
