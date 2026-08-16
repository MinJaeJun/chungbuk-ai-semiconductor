/* ==========================================================================
   xai.js - Global explainability tab
   ========================================================================== */
'use strict';

const Xai = {
  async init() {
    const res = await API.get('/api/xai/global');
    APP.xai = res;
    const order = ['xj_final_um', 'rsh_final_ohm_sq', 'xj_implant_um'];
    const color = { xj_final_um: 'purple', rsh_final_ohm_sq: 'cyan', xj_implant_um: '' };

    document.getElementById('xaiGlobal').innerHTML = order.map(t => {
      const b = res.targets[t];
      const perm = b.permutation.map(p => Object.assign({}, p, {
        numText: `ΔR² = ${fmt(p.value, 3)} (${p.share_pct.toFixed(1)}%)`,
      }));
      const shap = (b.shap_mean_abs || []).map(p => Object.assign({}, p, {
        numText: `mean|φ| = ${fmt(p.value, 3)} ${TARGET_UNIT[t]} (${p.share_pct.toFixed(1)}%)`,
      }));
      return `
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">${TARGET_LABEL[t]} Influence</span>
          <span class="panel-badge ${color[t]}">${esc(b.best_model)}</span>
        </div>
        <div class="panel-content">
          <div class="param-section-title">Permutation Importance (hold-out test)</div>
          ${barRows(perm, i => i.abs_value, i => `${i.label_ko} · ${i.label}`, 'pos')}
          ${shap.length ? `<div class="param-section-title mt16">Global SHAP · mean |φ|</div>${barRows(shap, i => i.abs_value, i => `${i.label_ko} · ${i.label}`, 'n1')}` : ''}
          <div class="param-hint mt12">단위: Permutation은 R² 감소량, SHAP은 ${TARGET_UNIT[t]}.</div>
        </div>
      </div>`;
    }).join('');

    // comparison chart
    document.getElementById('xaiMethod').innerHTML = `
      <div class="grid-2">
        <div>
          ${metricRows([
            ['Permutation Importance', esc(res.method.permutation)],
            ['SHAP', esc(res.method.shap)],
          ])}
          <div class="param-hint mt12">
            SHAP 라이브러리 의존 없이 4개 feature 전체 조합(2⁴ = 16 coalition)을 직접 평가하여
            <b>근사가 아닌 정확한 interventional Shapley value</b>를 계산합니다.
            efficiency 공리 Σφ = f(x) − E[f(X)]가 성립하는지 예측 탭에서 잔차로 확인할 수 있습니다.
          </div>
        </div>
        <div class="chart-box"><canvas id="xaiCompare"></canvas></div>
      </div>`;

    const feats = res.targets.xj_final_um.permutation.map(p => p.label);
    const share = (t, kind) => {
      const arr = res.targets[t][kind];
      const map = {}; arr.forEach(a => { map[a.label] = a.share_pct; });
      return feats.map(f => map[f] ?? 0);
    };
    drawChart('xaiCompare', {
      type: 'radar',
      data: {
        labels: feats,
        datasets: order.map((t, i) => ({
          label: TARGET_LABEL[t],
          data: share(t, 'permutation'),
          borderColor: SERIES[i], backgroundColor: hexA(SERIES[i], 0.14),
          borderWidth: 2, pointRadius: 3,
        })),
      },
      options: {
        plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.r.toFixed(1)}%` } } },
        scales: {
          r: {
            angleLines: { color: COLORS.grid }, grid: { color: COLORS.grid },
            pointLabels: { color: COLORS.text, font: { size: 10.5 } },
            ticks: { color: COLORS.text, backdropColor: 'transparent', font: { size: 9 } },
            suggestedMin: 0,
          },
        },
      },
    });
  },
};
