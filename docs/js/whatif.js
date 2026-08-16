/* ==========================================================================
   whatif.js - What-If / Process Window analysis
   ========================================================================== */
'use strict';

const WhatIf = {
  param: 'anneal_temp_C',
  nPoints: 61,

  init() {
    document.getElementById('wiParam').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#wiParam button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      WhatIf.param = b.dataset.p;
      WhatIf.run();
    });
    const sN = document.getElementById('sWiN');
    sN.addEventListener('input', () => {
      WhatIf.nPoints = parseInt(sN.value, 10);
      document.getElementById('vWiN').textContent = WhatIf.nPoints;
    });
    document.getElementById('btnSweep').addEventListener('click', () => WhatIf.run());
    WhatIf.renderBase();
  },

  renderBase() {
    const el = document.getElementById('wiBase');
    if (!el) return;
    el.innerHTML = metricRows([
      ['Implant Dose', fmtSci(APP.state.dose, 3) + ' cm⁻²'],
      ['Implant Energy', fmt(APP.state.energy, 4) + ' keV'],
      ['Anneal Temperature', fmt(APP.state.temp, 5) + ' °C'],
      ['Anneal Time', fmt(APP.state.time, 4) + ' sec'],
    ]) + '<div class="param-hint mt8">AI Prediction 탭에서 조건을 변경하면 여기에도 즉시 반영됩니다.</div>';
  },

  body(param, n) {
    return {
      parameter: param, n_points: n,
      dose_cm2: APP.state.dose, energy_keV: APP.state.energy,
      anneal_temp_C: APP.state.temp, anneal_time_sec: APP.state.time,
    };
  },

  async run() {
    const btn = document.getElementById('btnSweep');
    btn.disabled = true;
    WhatIf.renderBase();
    try {
      const res = await API.post('/api/whatif', WhatIf.body(WhatIf.param, WhatIf.nPoints));
      APP.state.lastSweep = res;
      WhatIf.render(res);
      await WhatIf.renderAll();
    } catch (e) {
      document.getElementById('wiSummary').innerHTML =
        `<div class="banner error"><span class="banner-icon">⚠</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false;
    }
  },

  render(res) {
    const p = res.parameter;
    const label = `${res.parameter_label} [${PARAM_UNIT[p]}]`;
    document.getElementById('wiT1').textContent = res.parameter_label;
    document.getElementById('wiT2').textContent = res.parameter_label;
    document.getElementById('wiBadge1').textContent = `${res.range.n_points} pts · ${fmt(res.range.min, 3)} → ${fmt(res.range.max, 3)}`;
    document.getElementById('wiBadge2').textContent = res.doe_reference.available
      ? `TCAD reference ${res.doe_reference.n_points} pts` : 'surrogate only';

    const xy = (arr) => res.x.map((x, i) => ({ x, y: arr[i] }));
    const refXy = (arr) => res.doe_reference.x.map((x, i) => ({ x, y: arr[i] }));
    const logx = p === 'dose_cm2';

    // Chart.js 'scatter' datasets do not draw a connecting line unless
    // showLine is set explicitly - without it the AI curves are invisible.
    const ds1 = [
      { label: 'AI · Xj Final', data: xy(res.curves.xj_final_um), borderColor: COLORS.purple, backgroundColor: COLORS.purple, pointRadius: 0, borderWidth: 2.4, tension: 0.2, showLine: true },
      { label: 'AI · Xj Implant', data: xy(res.curves.xj_implant_um), borderColor: COLORS.blue, backgroundColor: COLORS.blue, pointRadius: 0, borderWidth: 2, borderDash: [5, 4], tension: 0.2, showLine: true },
    ];
    if (res.doe_reference.available) {
      ds1.push({ label: 'TCAD · Xj Final (실제)', data: refXy(res.doe_reference.xj_final_um), showLine: false, pointRadius: 5, pointStyle: 'circle', borderColor: '#fff', backgroundColor: 'rgba(163,113,247,.95)', borderWidth: 1.5 });
    }
    drawChart('wiXjChart', {
      type: 'scatter',
      data: { datasets: ds1 },
      options: {
        plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 5)} um` } } },
        scales: { x: Object.assign(axis(label), logx ? { type: 'logarithmic' } : {}), y: axis('Junction Depth [um]') },
      },
    });

    const ds2 = [
      { label: 'AI · Rsh', data: xy(res.curves.rsh_final_ohm_sq), borderColor: COLORS.cyan, backgroundColor: COLORS.cyan, pointRadius: 0, borderWidth: 2.4, tension: 0.2, showLine: true },
    ];
    if (res.doe_reference.available) {
      ds2.push({ label: 'TCAD · Rsh (실제)', data: refXy(res.doe_reference.rsh_final_ohm_sq), showLine: false, pointRadius: 5, borderColor: '#fff', backgroundColor: 'rgba(57,211,83,.95)', borderWidth: 1.5 });
    }
    drawChart('wiRshChart', {
      type: 'scatter',
      data: { datasets: ds2 },
      options: {
        plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 5)} ohm/sq` } } },
        scales: { x: Object.assign(axis(label), logx ? { type: 'logarithmic' } : {}), y: axis('Sheet Resistance [ohm/sq]') },
      },
    });

    const s = res.summary;
    const dirKo = { increasing: '증가', decreasing: '감소', flat: '변화 없음' };
    document.getElementById('wiSummary').innerHTML = metricRows([
      ['Sweep 범위', `${fmt(res.range.min, 4)} → ${fmt(res.range.max, 4)} ${PARAM_UNIT[p]}`],
      ['Xj Final 변화', `${fmt(s.xj_final_um.at_start, 5)} → ${fmt(s.xj_final_um.at_end, 5)} um (${dirKo[s.xj_final_um.direction]})`],
      ['Xj Final 변화폭', `${fmt(s.xj_final_um.span, 3)} um`],
      ['ΔXj 변화폭', `${fmt(s.delta_xj_um.span, 3)} um`],
      ['Rsh 변화', `${fmt(s.rsh_final_ohm_sq.at_start, 5)} → ${fmt(s.rsh_final_ohm_sq.at_end, 5)} ohm/sq (${dirKo[s.rsh_final_ohm_sq.direction]})`],
      ['Rsh 변화폭', `${fmt(s.rsh_final_ohm_sq.span, 4)} ohm/sq`],
      ['TCAD reference', res.doe_reference.available ? `${res.doe_reference.n_points} runs 일치` : '일치 조건 없음'],
    ]) + `<div class="param-hint mt8">${res.doe_reference.available
      ? '나머지 3개 변수가 DOE 격자와 일치하여 실제 TCAD run을 겹쳐 표시했습니다. 곡선과 점의 일치도가 surrogate 신뢰도의 직접 증거입니다.'
      : '나머지 3개 변수가 DOE 격자점이 아니므로 비교할 실제 TCAD run이 없습니다. 결과는 AI 보간입니다.'}</div>`;
  },

  async renderAll() {
    const params = ['dose_cm2', 'energy_keV', 'anneal_temp_C', 'anneal_time_sec'];
    const results = await Promise.all(params.map(p => API.post('/api/whatif', WhatIf.body(p, 41))));
    results.forEach((res, i) => {
      const p = res.parameter;
      const ds = [
        { label: 'Xj Final [um]', data: res.x.map((x, k) => ({ x, y: res.curves.xj_final_um[k] })), borderColor: COLORS.purple, pointRadius: 0, borderWidth: 2, yAxisID: 'y', tension: 0.2, showLine: true },
        { label: 'Rsh [ohm/sq]', data: res.x.map((x, k) => ({ x, y: res.curves.rsh_final_ohm_sq[k] })), borderColor: COLORS.cyan, pointRadius: 0, borderWidth: 2, yAxisID: 'y1', tension: 0.2, showLine: true },
      ];
      if (res.doe_reference.available) {
        ds.push({ label: 'TCAD Xj', data: res.doe_reference.x.map((x, k) => ({ x, y: res.doe_reference.xj_final_um[k] })), showLine: false, pointRadius: 3.5, backgroundColor: 'rgba(163,113,247,.9)', borderWidth: 0, yAxisID: 'y' });
        ds.push({ label: 'TCAD Rsh', data: res.doe_reference.x.map((x, k) => ({ x, y: res.doe_reference.rsh_final_ohm_sq[k] })), showLine: false, pointRadius: 3.5, backgroundColor: 'rgba(57,211,83,.9)', borderWidth: 0, yAxisID: 'y1' });
      }
      drawChart(`wiAll${i}`, {
        type: 'scatter',
        data: { datasets: ds },
        options: {
          plugins: {
            legend: { display: false },
            title: { display: true, text: `${res.parameter_label} sweep`, color: '#8b949e', font: { size: 11 } },
          },
          scales: {
            x: Object.assign(axis(PARAM_UNIT[p]), p === 'dose_cm2' ? { type: 'logarithmic' } : {}),
            y: axis('Xj [um]', { position: 'left' }),
            y1: axis('Rsh', { position: 'right', grid: { drawOnChartArea: false } }),
          },
        },
      });
    });
  },
};
