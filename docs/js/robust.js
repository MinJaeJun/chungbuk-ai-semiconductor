/* ==========================================================================
   robust.js - Process Robustness tab
   Monte-Carlo spec yield under equipment scatter measured on a real line.
   ========================================================================== */
'use strict';

const Robust = {
  preset: 'typical',
  cv: null,
  variation: null,

  async init() {
    const tb = APP.meta.dataset.target_bounds;

    const sXj = document.getElementById('sRbXj');
    sXj.min = tb.xj_final_um.min; sXj.max = tb.xj_final_um.max;
    const sTol = document.getElementById('sRbTol');
    const sRsh = document.getElementById('sRbRsh');
    sRsh.min = Math.floor(tb.rsh_final_ohm_sq.min);
    sRsh.max = Math.ceil(tb.rsh_final_ohm_sq.max);

    // Centre the spec on what the current setpoint actually produces. Opening
    // on an arbitrary target would show a ~0% yield that says nothing about
    // robustness - only that the target and the recipe disagree.
    let nom = APP.state.lastPredict;
    if (!nom) {
      try {
        nom = await API.post('/api/predict', {
          dose_cm2: APP.state.dose, energy_keV: APP.state.energy,
          anneal_temp_C: APP.state.temp, anneal_time_sec: APP.state.time, explain: false,
        });
      } catch (_) { nom = null; }
    }
    sXj.value = nom ? nom.prediction.xj_final_um : 0.25;
    sRsh.value = nom ? Math.ceil(nom.prediction.rsh_final_ohm_sq * 1.05)
                     : Math.round(tb.rsh_final_ohm_sq.min * 2);

    const sync = () => {
      document.getElementById('vRbXj').textContent = fmtFixed(parseFloat(sXj.value), 4);
      document.getElementById('vRbTol').textContent = fmtFixed(parseFloat(sTol.value), 4);
      document.getElementById('vRbRsh').textContent = sRsh.value;
    };
    [sXj, sTol, sRsh].forEach(s => s.addEventListener('input', sync));
    sync();

    const v = await API.get('/api/variation');
    Robust.variation = v;
    Robust.renderVariation(v);

    // tolerance presets
    const presets = v.presets || {};
    document.getElementById('rbPreset').innerHTML = Object.entries(presets)
      .map(([k, p]) => `<button data-p="${k}"${k === Robust.preset ? ' class="active"' : ''}>${esc(p.label.split('·')[0].trim())}</button>`)
      .join('');
    document.getElementById('rbPreset').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#rbPreset button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      Robust.preset = b.dataset.p;
      Robust.cv = Object.assign({}, presets[Robust.preset].cv_pct);
      Robust.renderCv();
    });
    Robust.cv = Object.assign({}, (presets[Robust.preset] || {}).cv_pct || {});
    Robust.renderCv();

    document.getElementById('btnRobust').addEventListener('click', () => Robust.run());
    Robust.renderBase();
    await Robust.run();
  },

  renderBase() {
    const el = document.getElementById('rbBase');
    if (!el) return;
    el.innerHTML = metricRows([
      ['Implant Dose', fmtSci(APP.state.dose, 3) + ' cm⁻²'],
      ['Implant Energy', fmt(APP.state.energy, 4) + ' keV'],
      ['Anneal Temperature', fmt(APP.state.temp, 5) + ' °C'],
      ['Anneal Time', fmt(APP.state.time, 4) + ' sec'],
    ]);
  },

  renderVariation(v) {
    const box = document.getElementById('varRef');
    if (!v.available) {
      document.getElementById('varBadge').textContent = 'not available';
      box.innerHTML = `<div class="param-hint">${esc(v.detail || '실측 계측 데이터가 없습니다. 산포 tolerance는 사용자가 직접 지정해야 합니다.')}</div>`;
      return;
    }
    const s = v.source;
    document.getElementById('varBadge').textContent = `${fmtInt(s.rows)} rows · ${fmtInt(s.wafers)} wafers`;
    const ww = v.within_wafer.nu_pct, w2w = v.wafer_to_wafer.cv_pct, l2l = v.lot_to_lot.cv_pct;
    box.innerHTML = `
      <div class="grid-4 mb12">
        ${[
          ['WITHIN-WAFER NU', fmtFixed(ww.mean, 2), '%', `median ${fmtFixed(ww.median, 2)} · p90 ${fmtFixed(ww.p90, 2)} · ${s.points_per_wafer}점 맵`],
          ['WAFER-TO-WAFER CV', fmtFixed(w2w.mean, 3), '%', `median ${fmtFixed(w2w.median, 3)} · p90 ${fmtFixed(w2w.p90, 3)}`],
          ['LOT-TO-LOT CV', fmtFixed(l2l.mean, 3), '%', `${v.lot_to_lot.per_recipe.length} recipes`],
          ['POOLED TOTAL CV', fmtFixed(v.overall.total_cv_pct, 3), '%', `${fmtInt(s.wafers)} wafers 전체`],
        ].map(([l, val, u, sub], i) => `
          <div class="kpi-card ${['cyan', 'purple', 'yellow', ''][i]}">
            <div class="kpi-label">${l}</div>
            <div class="kpi-value">${val}<span class="kpi-unit">${u}</span></div>
            <div class="kpi-sub">${esc(sub)}</div>
          </div>`).join('')}
      </div>
      <div class="grid-2">
        <div>
          <div class="param-section-title">데이터 출처</div>
          ${metricRows([
            ['파일', esc(s.file)],
            ['공정', esc(s.process_ko)],
            ['규모', `${fmtInt(s.rows)} 계측 · ${s.lots} lot · ${fmtInt(s.wafers)} wafer · ${s.recipes} recipe`],
            ['기간', `${esc(String(s.date_from).slice(0, 16))} ~ ${esc(String(s.date_to).slice(0, 16))}`],
          ])}
        </div>
        <div>
          <div class="param-section-title">Recipe별 두께 · 균일도</div>
          <div class="table-wrap"><table class="grid-table">
            <thead><tr><th>Recipe</th><th>wafers</th><th>mean [Å]</th><th>std</th><th>within-wafer NU%</th></tr></thead>
            <tbody>${v.recipe_summary.map(r => `<tr>
              <td class="name">${esc(r.recipe)}</td><td>${r.wafers}</td>
              <td>${fmtFixed(r.mean, 1)}</td><td>${fmtFixed(r.std, 1)}</td>
              <td>${fmtFixed(r.within_wafer_nu_pct, 2)}</td></tr>`).join('')}</tbody>
          </table></div>
        </div>
      </div>
      <div class="banner warn mt12" style="margin-bottom:0">
        <span class="banner-icon">⚠</span>
        <div><b>범위 주의.</b> ${esc(v.caveat_ko)}<br>
        <span class="small mono">${esc(v.caveat_en)}</span></div>
      </div>`;
  },

  renderCv() {
    const el = document.getElementById('rbCvControls');
    el.innerHTML = Object.keys(Robust.cv).map(k => `
      <div class="param-group">
        <div class="param-label"><span>${PARAM_LABEL[k]}</span><span class="param-value" id="vcv_${k}">${fmtFixed(Robust.cv[k], 2)} %</span></div>
        <input type="range" class="param-slider" data-k="${k}" min="0" max="5" step="0.05" value="${Robust.cv[k]}">
      </div>`).join('')
      + `<div class="param-hint">각 공정 입력의 상대 1σ(%). 실측 라인 기준값에서 출발하지만 실제 이온주입기 tolerance로 교체해야 합니다.</div>`;
    el.querySelectorAll('input[type=range]').forEach(s => {
      s.addEventListener('input', () => {
        const k = s.dataset.k;
        Robust.cv[k] = parseFloat(s.value);
        document.getElementById(`vcv_${k}`).textContent = fmtFixed(Robust.cv[k], 2) + ' %';
      });
    });
  },

  body() {
    return {
      dose_cm2: APP.state.dose, energy_keV: APP.state.energy,
      anneal_temp_C: APP.state.temp, anneal_time_sec: APP.state.time,
      target_xj_um: parseFloat(document.getElementById('sRbXj').value),
      tolerance_um: parseFloat(document.getElementById('sRbTol').value),
      rsh_max: parseFloat(document.getElementById('sRbRsh').value),
      preset: Robust.preset,
      cv_pct: Robust.cv,
      n_samples: 2500,
    };
  },

  async run() {
    const btn = document.getElementById('btnRobust');
    btn.disabled = true;
    Robust.renderBase();
    document.getElementById('rbYield').innerHTML = '<div class="empty-state"><span class="loading"></span> Monte-Carlo 계산 중…</div>';
    const t0 = performance.now();
    try {
      const r = await API.post('/api/robust', Robust.body());
      APP.state.lastRobust = r;
      document.getElementById('rbTiming').textContent =
        `latency ${(performance.now() - t0).toFixed(0)} ms · ${fmtInt(r.n_samples)} samples`;
      Robust.render(r);
    } catch (e) {
      document.getElementById('rbYield').innerHTML =
        `<div class="banner error"><span class="banner-icon">⚠</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false;
    }
  },

  render(r) {
    const y = r.yield, cpk = r.cpk, sp = r.spec;

    // ---- boundary clipping warning (a real bias, not cosmetic) ----------
    const bc = r.boundary_clipping;
    document.getElementById('rbWarn').innerHTML = bc.significant
      ? `<div class="banner warn"><span class="banner-icon">⚠</span><div>
          <b>경계 clipping ${fmtFixed(bc.max_pct, 1)}%.</b> ${esc(bc.note_ko)}<br>
          <span class="small mono">${Object.entries(bc.pct_by_parameter).filter(([, v]) => v > 0.05)
            .map(([k, v]) => `${k} ${v.toFixed(1)}%`).join(' · ')}</span></div></div>`
      : '';

    // ---- yield cards ---------------------------------------------------
    const grade = (v) => (v >= 99.7 ? 'green' : v >= 95 ? 'yellow' : 'red');
    const cpkGrade = (c) => (c === null ? 'mut' : c >= 1.33 ? 'ok' : 'no');
    document.getElementById('rbYieldBadge').textContent = `${fmtInt(r.n_samples)} samples`;
    document.getElementById('rbYield').innerHTML = `
      <div class="results-grid">
        <div class="result-card hero">
          <div class="result-label">Joint Spec 수율 (Xj ∩ Rsh)</div>
          <div class="result-value" style="color:var(--accent-${grade(y.joint_pct)})">${fmtFixed(y.joint_pct, 2)}<span class="result-unit">%</span></div>
          <div class="result-sub">Xj ${fmtFixed(sp.lsl, 4)} ~ ${fmtFixed(sp.usl, 4)} um${sp.rsh_max !== null ? ` · Rsh ≤ ${fmt(sp.rsh_max, 4)}` : ''}</div>
        </div>
        <div class="result-card">
          <div class="result-label">Xj 수율</div>
          <div class="result-value">${fmtFixed(y.xj_pct, 2)}<span class="result-unit">%</span></div>
          <div class="result-sub">Cpk <span class="pill ${cpkGrade(cpk.xj)}">${cpk.xj === null ? '–' : fmtFixed(cpk.xj, 3)}</span></div>
        </div>
        <div class="result-card">
          <div class="result-label">Rsh 수율</div>
          <div class="result-value">${y.rsh_pct === null ? '–' : fmtFixed(y.rsh_pct, 2)}<span class="result-unit">%</span></div>
          <div class="result-sub">Cpk <span class="pill ${cpkGrade(cpk.rsh)}">${cpk.rsh === null ? '–' : fmtFixed(cpk.rsh, 3)}</span></div>
        </div>
      </div>
      <div class="metric-table mt12">
        <div class="metric-row"><span class="k">Xj 평균 ± 1σ</span><span class="v">${fmtFixed(r.distribution.xj_final_um.mean, 5)} ± ${fmt(r.distribution.xj_final_um.std, 3)}</span></div>
        <div class="metric-row"><span class="k">Xj 5~95%</span><span class="v">${fmtFixed(r.distribution.xj_final_um.p05, 5)} ~ ${fmtFixed(r.distribution.xj_final_um.p95, 5)}</span></div>
        <div class="metric-row"><span class="k">Rsh 평균 ± 1σ</span><span class="v">${fmtFixed(r.distribution.rsh_final_ohm_sq.mean, 4)} ± ${fmt(r.distribution.rsh_final_ohm_sq.std, 3)}</span></div>
        <div class="metric-row"><span class="k">Rsh 5~95%</span><span class="v">${fmtFixed(r.distribution.rsh_final_ohm_sq.p05, 4)} ~ ${fmtFixed(r.distribution.rsh_final_ohm_sq.p95, 4)}</span></div>
        <div class="metric-row"><span class="k">Tolerance 기준</span><span class="v">${esc(r.tolerances.label)}</span></div>
      </div>
      <div class="param-hint mt8">Cpk ≥ 1.33이 일반적인 공정 능력 합격선입니다. 위 수율은 <b>공정 산포만</b> 반영하며 계측 오차·장비 드리프트는 포함하지 않습니다.</div>`;

    // ---- histograms ----------------------------------------------------
    Robust.hist('rbXjHist', r.distribution.xj_final_um, 'Xj Final [um]', COLORS.purple,
      [{ v: sp.lsl, c: COLORS.red, t: 'LSL' }, { v: sp.usl, c: COLORS.red, t: 'USL' },
       { v: sp.target_xj_um, c: COLORS.blue, t: 'target' }]);
    document.getElementById('rbXjBadge').textContent =
      `σ = ${fmt(r.distribution.xj_final_um.std, 3)} um · CV ${fmtFixed(r.distribution.xj_final_um.cv_pct, 3)}%`;

    const rshLines = sp.rsh_max !== null ? [{ v: sp.rsh_max, c: COLORS.red, t: 'limit' }] : [];
    Robust.hist('rbRshHist', r.distribution.rsh_final_ohm_sq, 'Rsh [ohm/sq]', COLORS.cyan, rshLines);
    document.getElementById('rbRshBadge').textContent =
      `σ = ${fmt(r.distribution.rsh_final_ohm_sq.std, 3)} · CV ${fmtFixed(r.distribution.rsh_final_ohm_sq.cv_pct, 3)}%`;

    // ---- variance decomposition ----------------------------------------
    const dec = r.variance_decomposition;
    const rows = (key) => barRows(
      dec.slice().sort((a, b) => b[key] - a[key]).map(d => Object.assign({}, d, {
        numText: `${d[key].toFixed(1)}%  (입력 1σ ${d.cv_pct.toFixed(2)}%)`,
      })),
      i => i[key], i => PARAM_LABEL_KO[i.parameter] + ' · ' + PARAM_LABEL[i.parameter],
      key === 'xj_share_pct' ? 'pos' : 'n1');
    const topXj = dec.slice().sort((a, b) => b.xj_share_pct - a.xj_share_pct)[0];
    const topRsh = dec.slice().sort((a, b) => b.rsh_share_pct - a.rsh_share_pct)[0];
    document.getElementById('rbDecomp').innerHTML = `
      <div class="grid-2">
        <div><div class="param-section-title">Xj Final 산포 기여</div>${rows('xj_share_pct')}</div>
        <div><div class="param-section-title">Rsh 산포 기여</div>${rows('rsh_share_pct')}</div>
      </div>
      <div class="insight-card level-model mt12" style="margin-bottom:0">
        <span class="insight-level">MODEL INTERPRETATION</span>
        <div class="insight-text">
          현재 tolerance 설정에서 <b>Xj Final</b> 산포의 ${topXj.xj_share_pct.toFixed(1)}%는
          <b>${PARAM_LABEL_KO[topXj.parameter]}</b> 산포에서, <b>Rsh</b> 산포의 ${topRsh.rsh_share_pct.toFixed(1)}%는
          <b>${PARAM_LABEL_KO[topRsh.parameter]}</b> 산포에서 발생합니다.
          수율을 올리려면 다른 변수보다 이 변수의 장비 제어를 우선 개선하는 것이 효율적입니다.
          <span class="en">First-order variance decomposition: one input varied at a time, others held at setpoint.</span>
        </div>
      </div>`;

    // ---- model vs process uncertainty ----------------------------------
    const mu = r.model_uncertainty;
    document.getElementById('rbUnc').innerHTML = metricRows([
      ['① 모델 불확실성 · Xj', `${fmt(mu.xj_final_um.std, 3)} um`],
      ['① 모델 불확실성 · Rsh', `${fmt(mu.rsh_final_ohm_sq.std, 3)} ohm/sq`],
      ['② 공정 산포 · Xj (1σ)', `${fmt(r.distribution.xj_final_um.std, 3)} um`],
      ['② 공정 산포 · Rsh (1σ)', `${fmt(r.distribution.rsh_final_ohm_sq.std, 3)} ohm/sq`],
      ['비율 ②/① · Xj', mu.xj_final_um.std > 0 ? fmtInt(Math.round(r.distribution.xj_final_um.std / mu.xj_final_um.std)) + ' 배' : '–'],
    ]) + `<div class="param-hint mt8">
      <b>①</b> surrogate가 자기 답을 얼마나 확신하지 못하는가 (${esc(mu.xj_final_um.source)}).<br>
      <b>②</b> 장비 산포가 만들어내는 결과의 흩어짐.<br>
      ②가 ①보다 훨씬 크다는 것은 <b>지금 병목이 모델 정확도가 아니라 공정 제어</b>라는 뜻입니다.
      모델을 더 키우는 것보다 장비 tolerance를 줄이는 편이 수율에 직접 기여합니다.</div>`;
  },

  hist(canvasId, dist, label, color, lines) {
    const h = dist.histogram;
    drawChart(canvasId, {
      type: 'bar',
      data: {
        labels: h.centers.map(c => fmt(c, 5)),
        datasets: [{ label: 'Monte-Carlo 샘플 수', data: h.counts, backgroundColor: hexA(color, 0.65), borderWidth: 0, barPercentage: 1.0, categoryPercentage: 1.0 }],
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: c => `${label} ≈ ${c[0].label}`, label: c => `${c.parsed.y} samples` } },
          title: {
            display: true, color: '#6e7681', font: { size: 10.5 },
            text: lines.map(l => `${l.t} ${fmt(l.v, 5)}`).join('   |   '),
          },
        },
        scales: { x: axis(label, { ticks: { maxTicksLimit: 10, color: COLORS.text, font: { size: 9.5 } } }), y: axis('count') },
      },
      plugins: [{
        id: 'specLines',
        afterDraw(chart) {
          const { ctx, chartArea, scales } = chart;
          const lo = h.edges[0], hi = h.edges[h.edges.length - 1];
          lines.forEach(l => {
            if (l.v < lo || l.v > hi) return;
            const frac = (l.v - lo) / ((hi - lo) || 1);
            const x = chartArea.left + frac * (chartArea.right - chartArea.left);
            ctx.save();
            ctx.strokeStyle = l.c; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]);
            ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = l.c; ctx.font = '600 10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
            ctx.fillText(l.t, x, chartArea.top - 3);
            ctx.restore();
          });
          void scales;
        },
      }],
    });
  },
};
