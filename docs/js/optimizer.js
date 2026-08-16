/* ==========================================================================
   optimizer.js - AI Process Optimizer (multi-objective + Pareto)
   ========================================================================== */
'use strict';

const Optimizer = {
  mode: 'doe',
  rshMode: 'minimize',
  robust: false,

  init() {
    const tb = APP.meta.dataset.target_bounds;
    const lv = APP.meta.dataset.doe_levels;

    const sXj = document.getElementById('sOptXj');
    sXj.min = tb.xj_final_um.min; sXj.max = tb.xj_final_um.max;
    sXj.step = 0.001; sXj.value = 0.25;
    document.getElementById('optXjRange').textContent =
      `달성 가능 범위(TCAD DOE): ${fmtFixed(tb.xj_final_um.min, 4)} ~ ${fmtFixed(tb.xj_final_um.max, 4)} um`;
    sXj.addEventListener('input', () => {
      document.getElementById('vOptXj').textContent = fmtFixed(parseFloat(sXj.value), 4);
    });
    document.getElementById('vOptXj').textContent = fmtFixed(parseFloat(sXj.value), 4);

    const sTol = document.getElementById('sOptTol');
    sTol.addEventListener('input', () => {
      document.getElementById('vOptTol').textContent = fmtFixed(parseFloat(sTol.value), 3);
    });

    const sRsh = document.getElementById('sOptRsh');
    sRsh.min = Math.floor(tb.rsh_final_ohm_sq.min);
    sRsh.max = Math.ceil(tb.rsh_final_ohm_sq.max);
    sRsh.value = Math.round(tb.rsh_final_ohm_sq.min * 2);
    document.getElementById('vOptRsh').textContent = sRsh.value;
    sRsh.addEventListener('input', () => { document.getElementById('vOptRsh').textContent = sRsh.value; });

    const sW = document.getElementById('sOptW');
    const syncW = () => {
      const w = parseFloat(sW.value);
      document.getElementById('vOptW').textContent = `${w.toFixed(2)} / ${(1 - w).toFixed(2)}`;
    };
    sW.addEventListener('input', syncW); syncW();

    document.getElementById('optRshMode').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#optRshMode button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      Optimizer.rshMode = b.dataset.m;
      document.getElementById('rshMaxGroup').classList.toggle('hidden', b.dataset.m !== 'constraint');
    });

    document.getElementById('optMode').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#optMode button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      Optimizer.mode = b.dataset.m;
      document.getElementById('optModeHint').textContent = b.dataset.m === 'doe'
        ? '실제 TCAD 1,000 조건 중에서 탐색합니다. 추가 보간 오차가 없습니다.'
        : 'surrogate model로 DOE 격자 사이 조건까지 탐색합니다. 반드시 TCAD/Fab 검증이 필요합니다.';
    });

    const lockKey = document.getElementById('optLockKey');
    const lockVal = document.getElementById('optLockVal');
    lockKey.addEventListener('change', () => {
      const k = lockKey.value;
      document.getElementById('optLockValGroup').classList.toggle('hidden', !k);
      if (!k) return;
      lockVal.innerHTML = lv[k].map(v =>
        `<option value="${v}">${k === 'dose_cm2' ? fmtSci(v, 3) : fmt(v, 5)} ${PARAM_UNIT[k]}</option>`).join('');
    });

    document.getElementById('optRobust').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#optRobust button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      Optimizer.robust = b.dataset.r === '1';
    });

    document.getElementById('btnOptimize').addEventListener('click', () => Optimizer.run());
  },

  body() {
    const w = parseFloat(document.getElementById('sOptW').value);
    const lockKey = document.getElementById('optLockKey').value;
    const lockVal = document.getElementById('optLockVal').value;
    const body = {
      target_xj_um: parseFloat(document.getElementById('sOptXj').value),
      tolerance_um: parseFloat(document.getElementById('sOptTol').value),
      rsh_mode: Optimizer.rshMode,
      rsh_max: Optimizer.rshMode === 'constraint' ? parseFloat(document.getElementById('sOptRsh').value) : null,
      w_xj: w, w_rsh: 1 - w,
      mode: Optimizer.mode,
      top_k: 10,
      robust_rerank: Optimizer.robust,
      robust_preset: 'typical',
    };
    if (lockKey && lockVal) {
      body[{ dose_cm2: 'lock_dose_cm2', energy_keV: 'lock_energy_keV', anneal_temp_C: 'lock_anneal_temp_C', anneal_time_sec: 'lock_anneal_time_sec' }[lockKey]] = parseFloat(lockVal);
    }
    return body;
  },

  async run() {
    const btn = document.getElementById('btnOptimize');
    btn.disabled = true;
    document.getElementById('optSummary').innerHTML = '<div class="empty-state"><span class="loading"></span> 공정 조건 탐색 중…</div>';
    const t0 = performance.now();
    try {
      const res = await API.post('/api/optimize', Optimizer.body());
      APP.state.lastOptimize = res;
      document.getElementById('optTiming').textContent =
        `latency ${(performance.now() - t0).toFixed(0)} ms · ${fmtInt(res.search.candidates_evaluated)} candidates`;
      Optimizer.render(res);
    } catch (e) {
      document.getElementById('optSummary').innerHTML =
        `<div class="banner error"><span class="banner-icon">⚠</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false;
    }
  },

  render(res) {
    const obj = res.objective;

    document.getElementById('optBanner').innerHTML = res.verification_warning
      ? `<div class="banner warn"><span class="banner-icon">⚠</span><div><b>${esc(res.mode_label)}</b><br>
          ${esc(res.verification_warning)}<br><span class="small">${esc(res.verification_warning_ko)}</span></div></div>`
      : `<div class="banner ok"><span class="banner-icon">✔</span><div><b>${esc(res.mode_label)}</b><br>
          추천 조건의 Xj / Rsh 값은 실제 TCAD 시뮬레이션 결과입니다 (AI 예측값이 아님). 아래 <span class="mono">AI check</span> 열은 동일 조건에 대한 surrogate 예측으로, 모델 정확도 확인용입니다.</div></div>`;

    // recipes table
    document.getElementById('recipeBadge').textContent =
      `${res.recipes.length} recipes · ${fmtInt(res.search.feasible_count)} feasible`;
    const isDoe = res.mode === 'doe';
    document.getElementById('recipeTable').innerHTML = `
      <thead><tr>
        <th>Rank</th><th>Dose [cm⁻²]</th><th>Energy [keV]</th><th>Temp [°C]</th><th>Time [sec]</th>
        <th>${isDoe ? 'TCAD' : 'AI'} Xj Final [um]</th><th>Xj err [um]</th>
        <th>${isDoe ? 'TCAD' : 'AI'} Rsh [ohm/sq]</th><th>ΔXj [um]</th><th>Score</th><th>Spec</th>
        ${isDoe ? '<th>run_id</th><th>AI check (Xj / Rsh)</th>' : ''}
      </tr></thead>
      <tbody>${res.recipes.map(r => `
        <tr>
          <td><span class="rank-pill ${r.rank <= 3 ? 'top' : ''}">${r.rank}</span></td>
          <td>${fmtSci(r.recipe.dose_cm2, 3)}</td>
          <td>${fmt(r.recipe.energy_keV, 4)}</td>
          <td>${fmt(r.recipe.anneal_temp_C, 5)}</td>
          <td>${fmt(r.recipe.anneal_time_sec, 4)}</td>
          <td>${fmtFixed(r.predicted.xj_final_um, 5)}</td>
          <td>${fmt(r.xj_error_um, 3)}</td>
          <td>${fmtFixed(r.predicted.rsh_final_ohm_sq, 3)}</td>
          <td>${fmt(r.predicted.delta_xj_um, 3)}</td>
          <td>${fmtFixed(r.score, 4)}</td>
          <td><span class="pill ${r.feasible ? 'ok' : 'no'}">${r.feasible ? 'PASS' : 'OUT'}</span></td>
          ${isDoe ? `<td class="muted">${r.run_id}</td><td class="muted">${fmt(r.surrogate_check.xj_abs_error_um, 2)} / ${fmt(r.surrogate_check.rsh_abs_error_ohm_sq, 3)}</td>` : ''}
        </tr>`).join('')}
      </tbody>`;

    // Pareto chart
    document.getElementById('paretoBadge').textContent = `${res.pareto.length} non-dominated`;
    const cloud = res.cloud.xj_error_um.map((x, i) => ({ x, y: res.cloud.rsh_final_ohm_sq[i] }));
    const pareto = res.pareto.map(p => ({ x: p.xj_error_um, y: p.rsh_final_ohm_sq }));
    const tops = res.recipes.slice(0, 5).map(r => ({ x: r.xj_error_um, y: r.predicted.rsh_final_ohm_sq, r }));
    drawChart('paretoChart', {
      type: 'scatter',
      data: {
        datasets: [
          { label: '후보 조건 (score 상위)', data: cloud, pointRadius: 2.2, backgroundColor: 'rgba(110,118,129,.34)', borderWidth: 0 },
          { label: 'Pareto frontier', data: pareto, pointRadius: 4.5, backgroundColor: 'rgba(88,166,255,.95)', borderColor: '#fff', borderWidth: 1, showLine: true, tension: 0, borderDash: [], fill: false },
          { label: 'Top 5 추천', data: tops, pointRadius: 7, pointStyle: 'triangle', backgroundColor: 'rgba(210,153,34,.95)', borderColor: '#fff', borderWidth: 1.2 },
        ],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
          tooltip: {
            callbacks: {
              label: c => {
                const r = c.raw.r;
                if (r) return [`Rank ${r.rank}`, `Dose ${fmtSci(r.recipe.dose_cm2, 3)} · ${fmt(r.recipe.energy_keV, 4)} keV`,
                  `${fmt(r.recipe.anneal_temp_C, 5)} °C · ${fmt(r.recipe.anneal_time_sec, 4)} sec`,
                  `Xj err ${fmt(c.parsed.x, 3)} um · Rsh ${fmt(c.parsed.y, 4)}`];
                return `Xj err ${fmt(c.parsed.x, 3)} um · Rsh ${fmt(c.parsed.y, 4)} ohm/sq`;
              },
            },
          },
          title: {
            display: true, color: '#6e7681', font: { size: 10.5 },
            text: `목표 Xj = ${obj.target_xj_um} ± ${obj.tolerance_um} um · w_xj=${obj.w_xj.toFixed(2)}, w_rsh=${obj.w_rsh.toFixed(2)}${obj.constraint_note ? ' · ' + obj.constraint_note : ''}`,
          },
        },
        scales: {
          x: axis('|Xj − Xj_target| [um]  (작을수록 좋음)'),
          y: axis('Sheet Resistance [ohm/sq]  (작을수록 좋음)'),
        },
      },
    });

    const ar = res.achievable_range;
    document.getElementById('optSummary').innerHTML = metricRows([
      ['Search mode', res.mode_label.replace('MODE ', '')],
      ['평가 후보 수', fmtInt(res.search.candidates_evaluated)],
      ['Spec 만족 후보', fmtInt(res.search.feasible_count)],
      ['Pareto 후보', res.pareto.length],
      ['목표 Xj', `${obj.target_xj_um} ± ${obj.tolerance_um} um`],
      ['Rsh 조건', obj.rsh_mode === 'minimize' ? '최소화' : obj.constraint_note],
      ['가중치', `w_xj ${obj.w_xj.toFixed(2)} / w_rsh ${obj.w_rsh.toFixed(2)}`],
      ['탐색 공간 Xj', `${fmtFixed(ar.xj_final_um[0], 4)} ~ ${fmtFixed(ar.xj_final_um[1], 4)} um`],
      ['탐색 공간 Rsh', `${fmtFixed(ar.rsh_final_ohm_sq[0], 3)} ~ ${fmtFixed(ar.rsh_final_ohm_sq[1], 3)} ohm/sq`],
      ['Lock', Object.keys(res.search.locks).length ? Object.entries(res.search.locks).map(([k, v]) => `${PARAM_LABEL[k]}=${fmt(v, 4)}`).join(', ') : '없음'],
    ]) + `<div class="param-hint mt8">${esc(obj.formula)}</div>`;

    document.getElementById('optInsight').innerHTML = insightCards(res.insights);
    Optimizer.renderRobust(res);
  },

  renderRobust(res) {
    const panel = document.getElementById('robustPanel');
    if (!res.robust) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    const rows = res.robust.ranking;
    const moved = rows.filter(r => r.rank_shift !== 0).length;
    document.getElementById('robustBadge').textContent =
      `${esc(res.robust.tolerances.label)} · 순위 변동 ${moved}건`;
    document.getElementById('robustTable').innerHTML = `
      <thead><tr>
        <th>Robust</th><th>Nominal</th><th>변동</th>
        <th>Dose [cm⁻²]</th><th>Energy</th><th>Temp</th><th>Time</th>
        <th>Spec 수율 [%]</th><th>Xj 평균</th><th>Xj σ</th><th>Cpk</th><th>Rsh 평균</th>
      </tr></thead>
      <tbody>${rows.map(r => {
        const shift = r.rank_shift;
        const badge = shift > 0 ? `<span class="pill ok">▲${shift}</span>`
          : shift < 0 ? `<span class="pill no">▼${-shift}</span>` : '<span class="pill mut">–</span>';
        const y = r.yield_joint_pct;
        return `<tr>
          <td><span class="rank-pill ${r.robust_rank <= 3 ? 'top' : ''}">${r.robust_rank}</span></td>
          <td class="muted">${r.nominal_rank}</td>
          <td>${badge}</td>
          <td>${fmtSci(r.recipe.dose_cm2, 3)}</td>
          <td>${fmt(r.recipe.energy_keV, 4)}</td>
          <td>${fmt(r.recipe.anneal_temp_C, 5)}</td>
          <td>${fmt(r.recipe.anneal_time_sec, 4)}</td>
          <td style="color:${y >= 99.7 ? 'var(--accent-green)' : y >= 95 ? 'var(--accent-yellow)' : 'var(--accent-red)'}">${fmtFixed(y, 2)}</td>
          <td>${fmtFixed(r.xj_mean, 5)}</td>
          <td>${fmt(r.xj_std, 3)}</td>
          <td>${r.cpk_xj === null ? '–' : fmtFixed(r.cpk_xj, 2)}</td>
          <td>${fmtFixed(r.rsh_mean, 3)}</td>
        </tr>`;
      }).join('')}</tbody>`;
    panel.querySelector('.panel-content').insertAdjacentHTML('beforeend',
      `<div class="param-hint mt8">${moved
        ? `nominal 최적과 robust 최적의 순위가 <b>${moved}건</b> 달라졌습니다. 목표값에 가장 정확히 맞는 조건이 반드시 산포에 가장 강건한 조건은 아닙니다.`
        : '이 tolerance 수준에서는 nominal 순위와 robust 순위가 일치합니다. tolerance를 키우면 차이가 드러날 수 있습니다.'}</div>`);
  },
};
