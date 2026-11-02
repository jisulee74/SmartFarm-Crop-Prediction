// Presentation only: read the verified bundle; never select a winner on Test.
import { METRICS, selectedPooled } from './presentation.js?v=20261027';

const MODELS = { poisson: 'Poisson', random_forest: 'Random Forest', catboost: 'CatBoost', mlp: 'MLP', tabm: 'TabM', tft: 'TFT' };
const LABELS = ['RMSE ↓', 'MAE ↓', 'R² ↑', 'CCC ↑'];
const number = value => Number.isFinite(value) ? value.toFixed(3) : '산출 불가';
const integer = value => Number.isInteger(value) ? value.toLocaleString('ko-KR') : '확인 필요';

function metricText(metric, count) {
  if (!metric || !Number.isFinite(metric.mean) || !(metric.valid_count > 0)) return '산출 불가';
  if (count === 1) return number(metric.mean);
  return `${number(metric.mean)} (±${number(metric.std)})` +
    (metric.valid_count !== count ? ` · 유효 ${metric.valid_count}/${count}시드` : '');
}

function trussRow(model, truss) {
  const matches = model?.per_truss?.filter(row => row.truss === truss) || [];
  return matches.length === 1 ? matches[0] : null;
}

function appendRow(body, values, selected = false) {
  const row = body.insertRow();
  if (selected) row.className = 'truss-selected-model';
  values.forEach(value => { row.insertCell().textContent = value; });
}

function renderSelection(section, spec, chosen, count, truss, noun) {
  const row = trussRow(chosen, truss);
  section.dataset.truss = String(truss);
  section.querySelector('[data-truss-title]').textContent = `${truss}화방 ${noun} · Test 성능`;
  section.querySelector('[data-truss-counts]').textContent = row
    ? `평가 ${integer(row.n)}건 · 정답 양수 ${integer(row.positive_n)}건 · 선정 모델 ${MODELS[chosen.model]} / ${chosen.groups}`
    : '이 화방의 검증된 평가 결과가 없습니다.';
  section.querySelectorAll('[data-truss-metric]').forEach(card => {
    const metric = row?.metrics?.[card.dataset.trussMetric];
    card.querySelector('strong').textContent = metricText(metric, count);
  });
  const body = section.querySelector('tbody');
  body.replaceChildren();
  const bars = section.querySelector('[data-truss-bars]');
  bars.replaceChildren();
  const chartRows = [];
  for (const model of spec.models) {
    const detail = trussRow(model, truss);
    const selected = model.model === chosen.model;
    appendRow(body, [MODELS[model.model] + (selected ? ' · 대표 모델' : ''), model.groups,
      integer(detail?.n), integer(detail?.positive_n),
      ...METRICS.map(key => metricText(detail?.metrics?.[key], count))], selected);
    chartRows.push({ label: MODELS[model.model], rmse: detail?.metrics?.rmse?.mean, selected });
  }
  for (const [key, label] of [['persistence', '현재값 유지'], ['zero', '항상 0']]) {
    const baseline = trussRow(spec.baselines?.[key], truss);
    appendRow(body, [label, '기준선', integer(baseline?.n), integer(baseline?.positive_n),
      ...METRICS.map(key => number(baseline?.[key]))]);
    if (key === 'persistence') chartRows.push({ label, rmse: baseline?.rmse, selected: false });
  }
  // A performance chart, not a fabricated observation/prediction time series.
  const maximum = Math.max(0, ...chartRows.map(item => Number.isFinite(item.rmse) ? item.rmse : 0));
  for (const item of chartRows) {
    const barRow = document.createElement('div'); barRow.className = 'truss-bar-row';
    const label = document.createElement('span'); label.textContent = item.label;
    const track = document.createElement('div'); track.className = 'truss-bar-track';
    const bar = document.createElement('div'); bar.className = 'truss-bar';
    if (item.selected) bar.classList.add('selected');
    bar.style.width = `${maximum > 0 && Number.isFinite(item.rmse) ? item.rmse / maximum * 100 : 0}%`;
    track.append(bar);
    const value = document.createElement('span'); value.textContent = number(item.rmse);
    barRow.append(label, track, value); bars.append(barRow);
  }
  section.dataset.trussStatus = row ? 'ready' : 'unavailable';
}

export function renderTrussExplorers(root, bundle) {
  root.querySelectorAll('[data-summary-panel]').forEach(panel => {
    const crop = panel.dataset.summaryPanel;
    const { spec, chosen, count } = selectedPooled(bundle, crop);
    if (!Number.isInteger(spec.limit) || spec.limit < 1 || spec.limit !== bundle.limits?.[crop]) {
      throw new Error('화방 범위를 확인할 수 없습니다.');
    }
    panel.querySelector('[data-truss-explorer]')?.remove();
    const noun = crop === 'tomato' ? '꽃수 (개화수)' : '착과수';
    const section = document.createElement('section');
    section.className = 'truss-explorer'; section.dataset.trussExplorer = crop;
    section.setAttribute('aria-label', '화방별 예측 성능');
    // Static markup only; all result values are inserted with textContent.
    section.innerHTML = `<div class="truss-selector-heading"><div><h3>화방별 예측 성능</h3>
      <p>예측 대상을 선택하면 해당 화방의 평가 결과를 확인할 수 있습니다.</p></div>
      <div class="truss-selector-field"><label>예측 대상 (화방)</label><select class="exp-select"></select></div></div>
      <div aria-live="polite" aria-atomic="true"><h4 data-truss-title></h4><p data-truss-counts></p>
      <div class="truss-metrics"></div></div>
      <p class="truss-note" data-truss-note></p>
      <h4>선택 화방의 모델별 RMSE 비교</h4><div data-truss-bars class="truss-bars"></div>
      <details class="truss-model-details"><summary>선택 화방의 6개 모델·기준선 지표 보기</summary>
      <div class="crop-table-scroll"><table><caption>각 모델의 Validation 선정 입력 조합으로 계산한 Test 결과</caption>
      <thead><tr><th>모델</th><th>선정 입력</th><th>평가 건수</th><th>양수 건수</th><th>RMSE</th><th>MAE</th><th>R²</th><th>CCC</th></tr></thead><tbody></tbody></table></div></details>`;
    const select = section.querySelector('select');
    select.id = `crop_target_select_${crop}`;
    section.querySelector('label').htmlFor = select.id;
    for (let truss = 1; truss <= spec.limit; truss++) {
      const option = document.createElement('option');
      option.value = String(truss); option.textContent = `${truss}화방 ${noun}`;
      select.append(option);
    }
    METRICS.forEach((key, index) => {
      const card = document.createElement('article'); card.dataset.trussMetric = key;
      const label = document.createElement('span'); label.textContent = LABELS[index];
      card.append(label, document.createElement('strong'));
      section.querySelector('.truss-metrics').append(card);
    });
    section.querySelector('[data-truss-note]').textContent =
      '화방 번호를 입력하는 작물별 공통 모델의 결과입니다. 화방을 바꿔도 Validation에서 선정한 대표 모델·입력 조합은 유지하며, Test 결과로 재선정하지 않습니다. 위 대표 성능은 전체 화방 pooled 지표입니다.' +
      (count > 1 ? ` 화방별 모델 지표는 ${count}시드 지표의 평균과 표본 표준편차이며, 건수는 시드 반복을 제외한 평가 사례 수입니다. 시드 예측값의 앙상블 성능이 아닙니다.` : ' 화방별 모델 지표는 1시드 Test 결과입니다.') +
      ' 정답 분산이 없거나 필요한 결과가 없으면 해당 지표는 산출 불가로 표시합니다.';
    select.addEventListener('change', () => renderSelection(section, spec, chosen, count, Number(select.value), noun));
    panel.querySelector('.crop-details').before(section);
    renderSelection(section, spec, chosen, count, 1, noun);
  });
  // Keep the older target-specific experiment available, with an explicit scope label.
  const legacyLabel = document.querySelector('label[for="exp_target_select"]');
  if (legacyLabel) legacyLabel.textContent = '2. 과거 예측 대상 (1~3화방·합계)';
  const legacyTitle = document.querySelector('#crop_advanced > summary');
  if (legacyTitle) legacyTitle.textContent = '과거 8개 타깃 실험 참고 · 1~3화방 및 합계 (현재 20/5화방 결과와 별도)';
}
