/* ==========================================================================
   predict.js - AI Prediction tab (prediction + local XAI + insight)
   ========================================================================== */
'use strict';

const Predict = {
  init() {
    const lv = APP.meta.dataset.doe_levels;
    const bd = APP.meta.dataset.input_bounds;
    // APP.state is seeded in main.js, so this tab can be opened at any time

    // dose slider is in log space
    const sDose = document.getElementById('sDose');
    sDose.min = 0; sDose.max = 1; sDose.step = 0.001;
    sDose.value = valueToLogSlider(APP.state.dose, bd.dose_cm2.min, bd.dose_cm2.max);

    const bind = (sliderId, key, valId, bounds, step) => {
      const s = document.getElementById(sliderId);
      if (key !== 'dose') { s.min = bounds.min; s.max = bounds.max; s.step = step; s.value = APP.state[key]; }
      s.addEventListener('input', () => {
        if (key === 'dose') {
          APP.state.dose = logSliderToValue(parseFloat(s.value), bd.dose_cm2.min, bd.dose_cm2.max);
        } else {
          APP.state[key] = parseFloat(s.value);
        }
        Predict.syncLabels();
      });
    };
    bind('sDose', 'dose', 'vDose', bd.dose_cm2);
    bind('sEnergy', 'energy', 'vEnergy', bd.energy_keV, 0.1);
    bind('sTemp', 'temp', 'vTemp', bd.anneal_temp_C, 1);
    bind('sTime', 'time', 'vTime', bd.anneal_time_sec, 0.5);

    const chips = (elId, levels, key, fmtFn) => {
      document.getElementById(elId).innerHTML = levels.map(v =>
        `<span class="chip" data-v="${v}">${fmtFn(v)}</span>`).join('');
      document.getElementById(elId).addEventListener('click', e => {
        const c = e.target.closest('.chip'); if (!c) return;
        APP.state[key] = parseFloat(c.dataset.v);
        Predict.syncSliders(); Predict.syncLabels();
      });
    };
    chips('chipDose', lv.dose_cm2, 'dose', v => fmtSci(v, 2));
    chips('chipEnergy', lv.energy_keV, 'energy', v => fmt(v, 4));
    chips('chipTemp', lv.anneal_temp_C, 'temp', v => fmt(v, 5));
    chips('chipTime', lv.anneal_time_sec, 'time', v => fmt(v, 3));

    document.getElementById('btnPredict').addEventListener('click', () => Predict.run());
    document.getElementById('btnResetPred').addEventListener('click', () => {
      APP.state.dose = nearest(lv.dose_cm2, 1.5e15);
      APP.state.energy = nearest(lv.energy_keV, 20);
      APP.state.temp = nearest(lv.anneal_temp_C, 1000);
      APP.state.time = nearest(lv.anneal_time_sec, 25);
      Predict.syncSliders(); Predict.syncLabels(); Predict.run();
    });
    document.getElementById('btnRandomDoe').addEventListener('click', () => {
      const pick = a => a[Math.floor(Math.random() * a.length)];
      APP.state.dose = pick(lv.dose_cm2); APP.state.energy = pick(lv.energy_keV);
      APP.state.temp = pick(lv.anneal_temp_C); APP.state.time = pick(lv.anneal_time_sec);
      Predict.syncSliders(); Predict.syncLabels(); Predict.run();
    });

    Predict.syncLabels();
    Predict.run();
  },

  syncSliders() {
    const bd = APP.meta.dataset.input_bounds;
    document.getElementById('sDose').value = valueToLogSlider(APP.state.dose, bd.dose_cm2.min, bd.dose_cm2.max);
    document.getElementById('sEnergy').value = APP.state.energy;
    document.getElementById('sTemp').value = APP.state.temp;
    document.getElementById('sTime').value = APP.state.time;
  },

  syncLabels() {
    const lv = APP.meta.dataset.doe_levels;
    document.getElementById('vDose').textContent = fmtSci(APP.state.dose, 3) + ' cm⁻²';
    document.getElementById('vEnergy').textContent = fmt(APP.state.energy, 4) + ' keV';
    document.getElementById('vTemp').textContent = fmt(APP.state.temp, 5) + ' °C';
    document.getElementById('vTime').textContent = fmt(APP.state.time, 4) + ' sec';
    const mark = (elId, key, levels) => {
      document.querySelectorAll(`#${elId} .chip`).forEach(c => {
        const v = parseFloat(c.dataset.v);
        c.classList.toggle('active', Math.abs(v - APP.state[key]) < Math.abs(v) * 1e-9 + 1e-9);
      });
    };
    mark('chipDose', 'dose', lv.dose_cm2); mark('chipEnergy', 'energy', lv.energy_keV);
    mark('chipTemp', 'temp', lv.anneal_temp_C); mark('chipTime', 'time', lv.anneal_time_sec);
    if (typeof WhatIf !== 'undefined' && WhatIf.renderBase) WhatIf.renderBase();
  },

  body() {
    return {
      dose_cm2: APP.state.dose, energy_keV: APP.state.energy,
      anneal_temp_C: APP.state.temp, anneal_time_sec: APP.state.time, explain: true,
    };
  },

  async run() {
    const btn = document.getElementById('btnPredict');
    const st = document.getElementById('predStatus');
    btn.disabled = true; st.textContent = 'Running…';
    document.getElementById('predResults').innerHTML = '<div class="empty-state"><span class="loading"></span> AI 예측 계산 중…</div>';
    const t0 = performance.now();
    try {
      const res = await API.post('/api/predict', Predict.body());
      APP.state.lastPredict = res;
      const ms = performance.now() - t0;
      document.getElementById('predTiming').textContent =
        `latency ${ms.toFixed(0)} ms · SHAP 16 coalitions × background 160`;
      st.textContent = 'Complete';
      Predict.render(res);
    } catch (e) {
      st.textContent = 'Error';
      document.getElementById('predResults').innerHTML =
        `<div class="banner error"><span class="banner-icon">⚠</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false;
    }
  },

  render(res) {
    const p = res.prediction, u = res.units, unc = res.uncertainty, ex = res.extrapolation;

    // banner
    const kind = { validated_doe_point: 'ok', interpolation: 'info', extrapolation: 'error' }[ex.level];
    const icon = { validated_doe_point: '✔', interpolation: 'ℹ', extrapolation: '⚠' }[ex.level];
    const title = { validated_doe_point: 'VALIDATED DOE POINT', interpolation: 'AI INTERPOLATION', extrapolation: 'EXTRAPOLATION WARNING' }[ex.level];
    document.getElementById('predBanner').innerHTML =
      `<div class="banner ${kind}"><span class="banner-icon">${icon}</span><div>
        <b>${title}</b> ${esc(ex.message_ko)}<br>
        <span class="small muted mono">${esc(ex.message_en)}</span>
        ${ex.outside_parameters.length ? `<br><span class="small mono">out of range: ${ex.outside_parameters.map(k => PARAM_LABEL[k]).join(', ')}</span>` : ''}
      </div></div>`;

    // results
    const card = (label, value, unit, sub, hero) => `
      <div class="result-card ${hero ? 'hero' : ''}">
        <div class="result-label">${label}</div>
        <div class="result-value">${value}<span class="result-unit">${unit}</span></div>
        <div class="result-sub">${sub}</div>
      </div>`;
    const dres = res.delta_resolution || {};
    document.getElementById('predResults').innerHTML =
      card('Predicted Final Junction Depth · Xj Final', fmtFixed(p.xj_final_um, 5), u.xj_final_um,
        `± ${fmt(unc.xj_final_um.test_rmse, 2)} (hold-out RMSE)`, true) +
      card('Xj Implant', fmtFixed(p.xj_implant_um, 5), u.xj_implant_um, `± ${fmt(unc.xj_implant_um.test_rmse, 2)}`) +
      card('ΔXj (derived)', fmt(p.delta_xj_um, 4), u.delta_xj_um,
        dres.resolved === false
          ? `⚠ 분해능(${fmt(dres.resolution_limit, 2)}) 미만`
          : `derived · RMSE ${fmt(dres.measured_rmse ?? dres.resolution_limit, 2)}`) +
      card('Sheet Resistance · Rsh', fmtFixed(p.rsh_final_ohm_sq, 3), u.rsh_final_ohm_sq,
        `± ${fmt(unc.rsh_final_ohm_sq.test_rmse, 3)} (hold-out RMSE)`) +
      card('Xj 증가율', fmtFixed(p.xj_implant_um > 0 ? (p.delta_xj_um / p.xj_implant_um) * 100 : 0, 3), '%',
        'anneal 확산에 의한 접합깊이 증가 비율');

    if (dres.resolved === false) {
      document.getElementById('predResults').insertAdjacentHTML('beforeend',
        `<div class="banner warn" style="grid-column:1/-1;margin:2px 0 0">
          <span class="banner-icon">⚠</span><div>${esc(dres.message_ko)}<br>
          <span class="small mono">${esc(dres.message_en)}</span></div></div>`);
    }

    // nearest DOE run
    const nr = res.nearest_doe_run;
    document.getElementById('nearBadge').textContent = nr.exact_match ? 'exact match' : `distance ${nr.distance.toFixed(4)}`;
    document.getElementById('nearestBox').innerHTML = metricRows([
      ['run_id', nr.run.run_id],
      ['Dose', fmtSci(nr.run.dose_cm2, 3) + ' cm⁻²'],
      ['Energy', fmt(nr.run.energy_keV, 4) + ' keV'],
      ['Anneal', `${fmt(nr.run.anneal_temp_C, 5)} °C / ${fmt(nr.run.anneal_time_sec, 3)} sec`],
      ['TCAD Xj Final', fmtFixed(nr.run.xj_final_um, 6) + ' um'],
      ['TCAD Rsh', fmtFixed(nr.run.rsh_final_ohm_sq, 3) + ' ohm/sq'],
      ['AI − TCAD (Xj)', fmt(p.xj_final_um - nr.run.xj_final_um, 3) + ' um'],
      ['AI − TCAD (Rsh)', fmt(p.rsh_final_ohm_sq - nr.run.rsh_final_ohm_sq, 3) + ' ohm/sq'],
    ]) + `<div class="param-hint mt8">${nr.exact_match
      ? '입력 조건이 실제 TCAD run과 일치하므로 위 TCAD 값이 정답(ground truth)입니다.'
      : 'DOE 공간에서 정규화 거리 기준으로 가장 가까운 실제 시뮬레이션 run입니다.'}</div>`;

    Predict.junction(p, dres);
    Predict.sensitivity(res.explanation);
    Predict.localXai(res.explanation);
    document.getElementById('insightBox').innerHTML = insightCards(res.insights);
    Predict.llm(res);
  },

  junction(p, dres) {
    const cv = document.getElementById('junctionCanvas');
    const w = cv.parentElement.clientWidth || 600, h = 250;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const bd = APP.meta.dataset.target_bounds.xj_final_um;
    const depthMax = bd.max * 1.25;
    // ΔXj is 2-4 orders of magnitude smaller than Xj, so the full-scale view
    // can never resolve it. A magnified inset carries the junction region.
    const insetW = Math.min(190, Math.max(140, w * 0.26));
    const padL = 58, padR = insetW + 34, padT = 26, padB = 30;
    const W = w - padL - padR, H = h - padT - padB;
    const yOf = d => padT + (d / depthMax) * H;

    // ---- full-scale cross-section -------------------------------------
    ctx.fillStyle = '#2d333b'; ctx.fillRect(padL, padT, W, H);
    const gi = ctx.createLinearGradient(0, padT, 0, yOf(p.xj_implant_um));
    gi.addColorStop(0, 'rgba(88,166,255,.85)'); gi.addColorStop(1, 'rgba(88,166,255,.18)');
    ctx.fillStyle = gi; ctx.fillRect(padL, padT, W, yOf(p.xj_implant_um) - padT);
    ctx.fillStyle = 'rgba(163,113,247,.55)';
    ctx.fillRect(padL, yOf(p.xj_implant_um), W, Math.max(1.5, yOf(p.xj_final_um) - yOf(p.xj_implant_um)));

    ctx.strokeStyle = '#d29922'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL + W, padT); ctx.stroke();

    // junction lines; labels are stacked when the two depths nearly coincide
    const yi = yOf(p.xj_implant_um), yf = yOf(p.xj_final_um);
    const collide = Math.abs(yf - yi) < 16;
    const line = (d, y, color, label, dash, labelY) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.setLineDash(dash || []);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.font = '600 11px JetBrains Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText(`${label} ${d.toFixed(5)} um`, padL + 8, labelY);
    };
    line(p.xj_implant_um, yi, '#58a6ff', 'Xj implant', [5, 4], collide ? yi - 7 : yi - 5);
    line(p.xj_final_um, yf, '#a371f7', 'Xj final', [], collide ? yf + 15 : yf - 5);

    // depth axis
    ctx.strokeStyle = 'rgba(110,118,129,.45)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + H); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const d = (depthMax / 5) * i;
      ctx.fillText(d.toFixed(2), padL - 7, yOf(d) + 3.5);
      ctx.strokeStyle = 'rgba(110,118,129,.18)';
      ctx.beginPath(); ctx.moveTo(padL, yOf(d)); ctx.lineTo(padL + W, yOf(d)); ctx.stroke();
    }
    ctx.textAlign = 'left'; ctx.fillStyle = '#6e7681';
    ctx.fillText('depth [um]  ·  full scale', 6, padT - 10);
    ctx.fillStyle = '#d29922'; ctx.fillText('Si surface', padL + 2, padT + 14);

    // ---- magnified junction inset -------------------------------------
    const ix = w - insetW - 14, iy = padT, iw = insetW, ih = H;
    const limit = (dres && dres.resolution_limit) || 0;
    const span = Math.max(Math.abs(p.delta_xj_um) * 3, limit * 3, 2e-4);
    const dLo = p.xj_implant_um - span * 0.45, dHi = p.xj_implant_um + span;
    const zy = d => iy + ((d - dLo) / ((dHi - dLo) || 1)) * ih;

    ctx.fillStyle = '#22272e'; ctx.fillRect(ix, iy, iw, ih);
    ctx.strokeStyle = 'rgba(110,118,129,.45)'; ctx.lineWidth = 1; ctx.strokeRect(ix, iy, iw, ih);
    ctx.fillStyle = 'rgba(88,166,255,.30)';
    ctx.fillRect(ix + 1, iy + 1, iw - 2, Math.max(0, Math.min(ih - 2, zy(p.xj_implant_um) - iy)));
    const dyTop = Math.min(zy(p.xj_implant_um), zy(p.xj_final_um));
    const dyBot = Math.max(zy(p.xj_implant_um), zy(p.xj_final_um));
    ctx.fillStyle = 'rgba(163,113,247,.55)';
    ctx.fillRect(ix + 1, dyTop, iw - 2, Math.max(1.5, dyBot - dyTop));

    // resolution-limit band, drawn around Xj implant
    if (limit > 0) {
      ctx.fillStyle = 'rgba(210,153,34,.13)';
      ctx.fillRect(ix + 1, zy(p.xj_implant_um - limit), iw - 2, zy(p.xj_implant_um + limit) - zy(p.xj_implant_um - limit));
    }
    const mark = (d, color, dash) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash || []);
      ctx.beginPath(); ctx.moveTo(ix, zy(d)); ctx.lineTo(ix + iw, zy(d)); ctx.stroke(); ctx.setLineDash([]);
    };
    mark(p.xj_implant_um, '#58a6ff', [5, 4]);
    mark(p.xj_final_um, '#a371f7', []);

    ctx.font = '9.5px JetBrains Mono, monospace'; ctx.textAlign = 'left';
    ctx.fillStyle = '#6e7681';
    ctx.fillText(`zoom ×${Math.round((depthMax / (dHi - dLo)))}`, ix, iy - 12);
    ctx.fillStyle = '#a371f7';
    ctx.fillText(`ΔXj ${p.delta_xj_um.toExponential(2)} um`, ix + 5, Math.min(ih + iy - 22, dyBot + 12));
    if (limit > 0) {
      ctx.fillStyle = '#d29922';
      ctx.fillText(`± resolution ${limit.toExponential(1)}`, ix + 5, iy + ih - 7);
    }
  },

  sensitivity(exp) {
    if (!exp || !exp.xj_final_um) return;
    const feats = exp.xj_final_um.sensitivity.map(s => s.feature);
    const labels = exp.xj_final_um.sensitivity.map(s => s.label);
    const xjSpan = exp.xj_final_um.sensitivity.map(s => s.value);
    const rshMap = {};
    exp.rsh_final_ohm_sq.sensitivity.forEach(s => { rshMap[s.feature] = s.value; });
    const rshSpan = feats.map(f => rshMap[f]);

    drawChart('sensChart', {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Xj Final 변화폭 [um]', data: xjSpan, backgroundColor: 'rgba(88,166,255,.75)', yAxisID: 'y', borderRadius: 3 },
          { label: 'Rsh 변화폭 [ohm/sq]', data: rshSpan, backgroundColor: 'rgba(163,113,247,.75)', yAxisID: 'y1', borderRadius: 3 },
        ],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
          tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.parsed.y)}` } },
          title: { display: true, text: '나머지 3개 변수 고정 · 해당 변수만 학습범위 전체 sweep 시 예측 변화폭', color: '#6e7681', font: { size: 10.5 } },
        },
        scales: {
          x: axis(''),
          y: axis('Xj Final span [um]', { position: 'left' }),
          y1: axis('Rsh span [ohm/sq]', { position: 'right', grid: { drawOnChartArea: false } }),
        },
      },
    });
  },

  localXai(exp) {
    if (!exp || !exp.xj_final_um) return;
    const block = (target, color) => {
      const b = exp[target];
      const items = b.shap.map(s => Object.assign({}, s, {
        numText: `φ = ${fmt(s.value, 3)} ${TARGET_UNIT[target]} · ${s.share_pct.toFixed(1)}%`,
      }));
      return `
        <div class="mb12">
          <div class="row mb8">
            <span class="panel-badge ${color}">${TARGET_LABEL[target]}</span>
            <span class="spacer"></span>
            <span class="small muted mono">f(x)=${fmt(b.prediction, 4)} · E[f(X)]=${fmt(b.base_value, 4)} · Σφ=${fmt(b.prediction - b.base_value, 3)}</span>
          </div>
          ${barRows(items, i => i.value, i => `${i.label_ko} (${i.label})`, 'auto')}
        </div>`;
    };
    document.getElementById('localXai').innerHTML =
      block('xj_final_um', 'purple') + block('rsh_final_ohm_sq', 'cyan') +
      `<div class="param-hint">파란색 = 데이터 평균 대비 예측을 증가시킨 기여, 주황색 = 감소시킨 기여.
       4개 변수 전체 조합(2⁴=16 coalition)으로 계산한 정확한 interventional Shapley value이며,
       efficiency 잔차는 ${fmt(exp.xj_final_um.efficiency_residual, 2)} (Xj) / ${fmt(exp.rsh_final_ohm_sq.efficiency_residual, 2)} (Rsh) 입니다.</div>`;
  },

  async llm(res) {
    const info = APP.meta.llm;
    const badge = document.getElementById('llmBadge');
    const box = document.getElementById('llmBox');
    if (!info || !info.enabled) {
      badge.textContent = 'disabled';
      box.innerHTML = `<div class="param-hint">외부 LLM API Key가 설정되지 않았습니다. 모든 핵심 기능(예측 · XAI · 최적화)은 LLM 없이 동작합니다.
        <br>선택 확장: 환경변수 <span class="mono">OPENAI_API_KEY</span> / <span class="mono">ANTHROPIC_API_KEY</span> / <span class="mono">GEMINI_API_KEY</span> 설정 후 서버를 재시작하면 위 수치를 자연어로 요약합니다.</div>`;
      return;
    }
    badge.textContent = info.provider;
    box.innerHTML = '<div class="empty-state"><span class="loading"></span> LLM 요약 생성 중…</div>';
    try {
      const out = await API.post('/api/insight/llm', {
        evidence: {
          input: res.input, prediction: res.prediction,
          extrapolation_level: res.extrapolation.level,
          shap: res.explanation,
          model_uncertainty: res.uncertainty,
        },
      });
      box.innerHTML = out.text
        ? `<div class="insight-card level-model"><span class="insight-level">LLM NARRATIVE (${esc(out.provider)})</span><div class="insight-text">${esc(out.text)}</div></div>`
        : `<div class="param-hint">LLM 호출 실패: ${esc(out.error || out.note || '-')}</div>`;
    } catch (e) {
      box.innerHTML = `<div class="param-hint">LLM 호출 실패: ${esc(e.message)}</div>`;
    }
  },
};
