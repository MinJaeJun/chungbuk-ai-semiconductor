/* ==========================================================================
   explorer.js - Data Explorer tab
   Everything rendered here comes straight from implant_anneal_1000.csv.
   ========================================================================== */
'use strict';

const Explorer = {
  init() {
    const v = APP.summary.validation;
    const inputs = APP.meta.dataset.inputs;
    const outs = ['xj_implant_um', 'xj_final_um', 'delta_xj_um', 'rsh_final_ohm_sq'];

    document.getElementById('dsBadge').textContent = `${fmtInt(v.n_rows)} runs · ${v.n_columns} columns`;
    document.getElementById('doeBadge').textContent = v.doe.structure;

    document.getElementById('storyFlow').innerHTML = [
      'TCAD Simulation', 'Structured DOE Data', 'AI Surrogate Model', 'Fast Prediction',
      'Explainable AI', 'Process Window', 'Multi-objective Optimization', 'Recommended Recipe',
      'Engineer Decision Support',
    ].map((n, i) => `<span class="node${i === 1 ? ' hi' : ''}">${n}</span>`)
      .join('<span class="arrow">→</span>');

    document.getElementById('dsStats').innerHTML = [
      ['TOTAL TCAD RUNS', fmtInt(v.n_rows), 'runs', v.doe.is_full_factorial ? 'Full factorial · balanced' : 'Non-factorial'],
      ['INPUT PARAMETERS', v.n_inputs, 'controllable', inputs.map(i => PARAM_LABEL[i]).join(', ')],
      ['OUTPUT METRICS', outs.length, 'metrics', 'Xj implant / Xj final / ΔXj / Rsh'],
      ['MISSING VALUES', v.total_missing, 'cells', `duplicate rows ${v.duplicate_rows} · duplicate conditions ${v.duplicate_conditions}`],
    ].map(([l, val, u, sub], i) => `
      <div class="kpi-card ${['', 'cyan', 'purple', v.total_missing === 0 ? 'green' : 'yellow'][i]}">
        <div class="kpi-label">${l}</div>
        <div class="kpi-value">${val}<span class="kpi-unit">${u}</span></div>
        <div class="kpi-sub">${esc(sub)}</div>
      </div>`).join('');

    // column table
    const roleTag = { process_input: 'INPUT', model_target: 'AI TARGET', derived_target: 'DERIVED', identifier: 'ID' };
    const rolePill = { process_input: 'ok', model_target: 'ok', derived_target: 'mut', identifier: 'no' };
    document.getElementById('colTable').innerHTML = `
      <thead><tr>
        <th>Column</th><th>Role</th><th>dtype</th><th>Unit</th><th>Unique</th>
        <th>Missing</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th>
      </tr></thead>
      <tbody>${v.columns.map(c => `
        <tr>
          <td>${esc(c.name)}</td>
          <td><span class="pill ${rolePill[c.role]}">${roleTag[c.role]}</span></td>
          <td class="muted">${esc(c.dtype)}</td>
          <td class="muted">${esc(c.unit)}</td>
          <td>${c.n_unique}</td>
          <td>${c.n_missing}</td>
          <td>${fmt(c.min)}</td><td>${fmt(c.max)}</td>
          <td>${fmt(c.mean)}</td><td>${fmt(c.std)}</td>
        </tr>`).join('')}
      </tbody>`;

    // DOE matrix
    document.getElementById('doeMatrix').innerHTML = `
      ${metricRows(inputs.map(k => [
        `${PARAM_LABEL[k]} (${v.doe.level_counts[k]} levels)`,
        v.doe.levels[k].map(x => (k === 'dose_cm2' ? fmtSci(x, 2) : fmt(x, 4))).join(' · '),
      ]))}
      <div class="metric-table mt12">
        <div class="metric-row"><span class="k">Design</span><span class="v">${esc(v.doe.structure)}</span></div>
        <div class="metric-row"><span class="k">Full factorial</span><span class="v">${v.doe.is_full_factorial ? '✔ 확인됨' : '✘'}</span></div>
        <div class="metric-row"><span class="k">${esc(v.delta_identity.expression)}</span><span class="v">max|res| = ${fmt(v.delta_identity.max_abs_residual)}</span></div>
      </div>
      <div class="banner info mt12" style="margin-bottom:0">
        <span class="banner-icon">🔒</span>
        <div><b>Leakage guard.</b> 모델 입력은 ${inputs.join(', ')} 4개만 사용합니다.
        제외: ${v.leakage_policy.excluded_from_features.join(', ')}.
        ΔXj는 학습 대상이 아니라 예측된 Xj_final − Xj_implant로 계산합니다.</div>
      </div>`;

    // selectors
    const xSel = document.getElementById('exX'), ySel = document.getElementById('exY'), cSel = document.getElementById('exC');
    xSel.innerHTML = inputs.map(i => `<option value="${i}">${PARAM_LABEL[i]} [${PARAM_UNIT[i]}]</option>`).join('');
    ySel.innerHTML = outs.map(o => `<option value="${o}">${TARGET_LABEL[o]} [${TARGET_UNIT[o]}]</option>`).join('');
    cSel.innerHTML = inputs.concat(outs).map(o =>
      `<option value="${o}">${PARAM_LABEL[o] || TARGET_LABEL[o]}</option>`).join('');
    xSel.value = 'dose_cm2'; ySel.value = 'rsh_final_ohm_sq'; cSel.value = 'energy_keV';
    [xSel, ySel, cSel].forEach(s => s.addEventListener('change', () => Explorer.scatter()));

    document.getElementById('corrToggle').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#corrToggle button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      document.getElementById('corrBadge').textContent = b.dataset.m === 'pearson' ? 'Pearson' : 'Spearman';
      Explorer.heatmap(b.dataset.m);
    });
    document.getElementById('pcColor').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#pcColor button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      Explorer.parallel(b.dataset.c);
    });

    Explorer.scatter();
    Explorer.marginals();
    Explorer.heatmap('pearson');
    Explorer.parallel('rsh_final_ohm_sq');
    Explorer.observations();
    window.addEventListener('resize', () => {
      clearTimeout(Explorer._rt);
      Explorer._rt = setTimeout(() => Explorer.parallel(Explorer._pcColor || 'rsh_final_ohm_sq'), 200);
    });
  },

  scatter() {
    const cols = APP.points.columns;
    const xk = document.getElementById('exX').value;
    const yk = document.getElementById('exY').value;
    const ck = document.getElementById('exC').value;
    const cv = cols[ck];
    const cmin = Math.min(...cv), cmax = Math.max(...cv);
    const data = cols[xk].map((x, i) => ({ x, y: cols[yk][i] }));
    const colors = cv.map(v => ramp((v - cmin) / ((cmax - cmin) || 1)));
    document.getElementById('scatterBadge').textContent =
      `${PARAM_LABEL[xk]} → ${TARGET_LABEL[yk]} · color: ${PARAM_LABEL[ck] || TARGET_LABEL[ck]}`;

    drawChart('exScatter', {
      type: 'scatter',
      data: { datasets: [{ label: `${TARGET_LABEL[yk]}`, data, pointBackgroundColor: colors, pointRadius: 3, pointHoverRadius: 5, borderWidth: 0 }] },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => `${PARAM_LABEL[xk]}: ${paramText(xk, c.parsed.x)} → ${TARGET_LABEL[yk]}: ${fmt(c.parsed.y)} ${TARGET_UNIT[yk]}`,
              afterLabel: (c) => `${PARAM_LABEL[ck] || TARGET_LABEL[ck]}: ${fmt(cv[c.dataIndex])}`,
            },
          },
        },
        scales: {
          x: Object.assign(axis(`${PARAM_LABEL[xk]} [${PARAM_UNIT[xk]}]`),
            xk === 'dose_cm2' ? { type: 'logarithmic' } : {}),
          y: axis(`${TARGET_LABEL[yk]} [${TARGET_UNIT[yk]}]`),
        },
      },
    });
  },

  marginals() {
    const me = APP.summary.marginal_effects;
    const cols = APP.points.columns;

    const build = (canvasId, param, target, color, logx) => {
      const levels = me[param].levels;
      const means = me[param][target];
      // min / max at each level from the raw data (real spread across the other 3 params)
      const lo = [], hi = [];
      levels.forEach(L => {
        const vals = cols[param].map((v, i) => (Math.abs(v - L) < 1e-9 ? cols[target][i] : null)).filter(v => v !== null);
        lo.push(Math.min(...vals)); hi.push(Math.max(...vals));
      });
      drawChart(canvasId, {
        type: 'line',
        data: {
          labels: levels.map(l => (logx ? fmtSci(l, 2) : fmt(l, 4))),
          datasets: [
            { label: 'max (다른 변수 조합)', data: hi, borderColor: 'rgba(110,118,129,.45)', borderDash: [4, 3], pointRadius: 0, fill: '+1', backgroundColor: 'rgba(88,166,255,.07)' },
            { label: 'min (다른 변수 조합)', data: lo, borderColor: 'rgba(110,118,129,.45)', borderDash: [4, 3], pointRadius: 0 },
            { label: `mean ${TARGET_LABEL[target]}`, data: means, borderColor: color, backgroundColor: color, pointRadius: 4, borderWidth: 2.4, tension: 0.25 },
          ],
        },
        options: {
          plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y)} ${TARGET_UNIT[target]}` } } },
          scales: {
            x: axis(`${PARAM_LABEL[param]} [${PARAM_UNIT[param]}]`),
            y: axis(`${TARGET_LABEL[target]} [${TARGET_UNIT[target]}]`),
          },
        },
      });
    };
    build('chDoseRsh', 'dose_cm2', 'rsh_final_ohm_sq', COLORS.blue, true);
    build('chEnergyXj', 'energy_keV', 'xj_final_um', COLORS.purple, false);
    build('chTempRsh', 'anneal_temp_C', 'rsh_final_ohm_sq', COLORS.cyan, false);
    build('chTimeXj', 'anneal_time_sec', 'delta_xj_um', COLORS.yellow, false);
  },

  heatmap(method) {
    const c = APP.summary.correlation;
    const m = c[method];
    const n = c.labels.length;
    const short = c.labels.map(l => l.replace('Implant ', '').replace('Anneal ', 'An.').replace('Temperature', 'Temp'));
    let html = `<div class="heatmap-grid" style="grid-template-columns:86px repeat(${n},1fr)">`;
    html += '<div></div>';
    short.forEach(l => { html += `<div class="heat-label top">${esc(l)}</div>`; });
    for (let i = 0; i < n; i++) {
      html += `<div class="heat-label">${esc(short[i])}</div>`;
      for (let j = 0; j < n; j++) {
        const v = m[i][j];
        html += `<div class="heat-cell" style="background:${divergent(v)}" title="${esc(short[i])} vs ${esc(short[j])} = ${v.toFixed(3)}">${v.toFixed(2)}</div>`;
      }
    }
    html += '</div>';
    document.getElementById('corrHeat').innerHTML = html;
  },

  parallel(colorKey) {
    Explorer._pcColor = colorKey;
    const cv = document.getElementById('pcCanvas');
    const cols = APP.points.columns;
    const axes = ['dose_cm2', 'energy_keV', 'anneal_temp_C', 'anneal_time_sec', 'xj_final_um', 'rsh_final_ohm_sq'];
    const w = cv.parentElement.clientWidth || 600;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = 300 * dpr;
    cv.style.width = w + 'px'; cv.style.height = '300px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, 300);

    const padL = 14, padR = 14, padT = 26, padB = 34;
    const H = 300 - padT - padB;
    const xs = axes.map((_, i) => padL + i * ((w - padL - padR) / (axes.length - 1)));
    const scaled = axes.map(k => {
      let arr = cols[k];
      if (k === 'dose_cm2') arr = arr.map(Math.log10);
      const mn = Math.min(...arr), mx = Math.max(...arr);
      return { arr, mn, mx };
    });
    const cvals = cols[colorKey];
    const cmin = Math.min(...cvals), cmax = Math.max(...cvals);

    // axes
    ctx.strokeStyle = 'rgba(110,118,129,.4)'; ctx.lineWidth = 1;
    ctx.font = '10px JetBrains Mono, monospace'; ctx.fillStyle = '#8b949e'; ctx.textAlign = 'center';
    xs.forEach((x, i) => {
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + H); ctx.stroke();
      const k = axes[i];
      ctx.fillText((PARAM_LABEL[k] || TARGET_LABEL[k]).replace('Implant ', '').replace('Anneal ', ''), x, 300 - 18);
      ctx.fillText(k === 'dose_cm2' ? 'log10' : PARAM_UNIT[k] || TARGET_UNIT[k], x, 300 - 6);
    });

    // polylines (every row, thin + transparent)
    const n = cols[axes[0]].length;
    for (let r = 0; r < n; r++) {
      const t = (cvals[r] - cmin) / ((cmax - cmin) || 1);
      ctx.strokeStyle = ramp(t).replace('rgb(', 'rgba(').replace(')', ',0.16)');
      ctx.beginPath();
      for (let i = 0; i < axes.length; i++) {
        const s = scaled[i];
        const y = padT + H - ((s.arr[r] - s.mn) / ((s.mx - s.mn) || 1)) * H;
        if (i === 0) ctx.moveTo(xs[i], y); else ctx.lineTo(xs[i], y);
      }
      ctx.stroke();
    }
    // colorbar
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6e7681';
    ctx.fillText(`color: ${TARGET_LABEL[colorKey]}  ${fmt(cmin)} → ${fmt(cmax)} ${TARGET_UNIT[colorKey]}`, padL, 14);
  },

  observations() {
    const obs = APP.summary.observations.slice(0, 8);
    document.getElementById('obsList').innerHTML = obs.map(o => `
      <div class="insight-card level-data">
        <span class="insight-level">DATA OBSERVATION</span>
        <div class="insight-title">${esc(PARAM_LABEL_KO[o.parameter])} → ${esc(TARGET_LABEL[o.target])} · 주효과 ${o.effect_span_pct.toFixed(1)}%</div>
        <div class="insight-text">${esc(o.text_ko)}<span class="en">${esc(o.text_en)}</span></div>
      </div>`).join('');
  },
};
