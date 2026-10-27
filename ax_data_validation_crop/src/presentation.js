// Display policy only. The validated experiment bundle and frozen selection stay unchanged.
export const METRICS = ['rmse', 'mae', 'r2', 'ccc'];
const MODEL = { poisson: 'Poisson', random_forest: 'Random Forest', catboost: 'CatBoost', mlp: 'MLP', tabm: 'TabM', tft: 'TFT' };
const EXPLANATION = '대표 성능은 Validation RMSE로 선정한 모델의 Test 예측값을 전체 평가 행 기준으로 통합하여 산출했습니다.';
const format = value => value == null ? '산출 불가' : value.toFixed(3);

export function selectedPooled(bundle, crop) {
  if (bundle?.schema !== 2 || bundle.selection !== 'validation pooled RMSE') throw new Error('검증된 Validation 선정 결과가 필요합니다.');
  const count = bundle.seed_count;
  if (!Number.isInteger(count) || count < 1 || bundle.seeds?.length !== count || new Set(bundle.seeds).size !== count ||
      !['quick', 'formal'].includes(bundle.mode) || (bundle.mode === 'formal' && count !== 3) || (bundle.mode === 'quick' && count !== 1)) {
    throw new Error('시드 구성 확인이 필요합니다.');
  }
  const spec = bundle.crops?.[crop];
  const matches = spec?.models?.filter(model => model.model === spec.selected) || [];
  if (matches.length !== 1 || !MODEL[spec.selected]) throw new Error('동결된 선정 모델을 확인할 수 없습니다.');
  const chosen = matches[0];
  for (const key of METRICS) {
    const metric = chosen.pooled?.[key];
    if (!metric || !Number.isInteger(metric.valid_count) || metric.valid_count < 0 || metric.valid_count > count) throw new Error('pooled Test 지표가 없습니다.');
    if (metric.mean === null && metric.valid_count === 0 && metric.std === null && ['r2', 'ccc'].includes(key)) continue;
    if (!Number.isFinite(metric.mean) || metric.valid_count !== count ||
        (count > 1 ? !Number.isFinite(metric.std) || metric.std < 0 : metric.std !== null)) {
      throw new Error('모든 시드의 pooled Test 지표를 확인할 수 없습니다.');
    }
  }
  return { chosen, spec, count };
}

function macroDetails(panel, chosen, count) {
  if (panel.querySelector('[data-macro-detail]')) return;
  const details = panel.querySelector('.crop-details');
  if (!details) return;
  const section = document.createElement('section');
  section.dataset.macroDetail = '';
  const title = document.createElement('h3');
  title.textContent = '화방별 지표 산술평균 · 과거 대표 집계';
  const note = document.createElement('p');
  note.textContent = '각 시드에서 화방별 지표를 동일 비중으로 평균한 값입니다. R² 등 정의되지 않는 화방은 해당 평균에서 제외합니다.' +
    (count > 1 ? ' 표는 시드 간 평균이며 괄호는 시드 표준편차입니다.' : ' 대표 pooled 지표와 집계 방식이 다릅니다.');
  const wrapper = document.createElement('div'); wrapper.className = 'crop-table-scroll';
  const table = document.createElement('table');
  const header = table.createTHead().insertRow();
  ['RMSE', 'MAE', 'R²', 'CCC'].forEach(label => { const th = document.createElement('th'); th.textContent = label; header.append(th); });
  const row = table.createTBody().insertRow();
  METRICS.forEach(key => {
    const metric = chosen.macro?.[key];
    row.insertCell().textContent = format(metric?.mean) + (metric?.std != null ? ` (±${format(metric.std)})` : '');
  });
  wrapper.append(table); section.append(title, note, wrapper); details.append(section);
}

function unavailable(panel, reason) {
  panel.querySelectorAll('.crop-metric strong').forEach(node => { node.textContent = '산출 불가'; });
  panel.querySelectorAll('.crop-metric small').forEach(node => { node.textContent = 'pooled Test 지표 확인 필요'; });
  const note = panel.querySelector('.crop-explanation');
  if (note) note.textContent = `대표 성능을 표시할 수 없습니다. ${reason} 화방 평균으로 대체하지 않습니다.`;
  panel.dataset.pooledStatus = 'unavailable';
}

export function showUnavailable(root, reason) {
  root.querySelectorAll('[data-summary-panel]').forEach(panel => unavailable(panel, reason));
  root.dataset.presentationReady = 'error';
}

export function renderPooled(root, bundle) {
  const status = root.querySelector('.crop-status')?.textContent || '';
  let success = true;
  root.querySelectorAll('[data-summary-panel]').forEach(panel => {
    try {
      const { chosen, spec, count } = selectedPooled(bundle, panel.dataset.summaryPanel);
      // Do not combine stale HTML from one release with another release's summary.
      const label = panel.querySelector(':scope > p')?.textContent || '';
      if (!status.includes(`${count}시드`) || !label.includes(`선정 모델: ${MODEL[chosen.model]} · 입력: ${chosen.groups} ·`) ||
          !label.includes(`${Number(spec.counts.test.rows).toLocaleString('en-US')}건`) ||
          !panel.querySelector('h2')?.textContent.includes(`1~${spec.limit}화방`)) {
        throw new Error('화면과 결과 버전이 다릅니다. 새로고침 후 확인하세요.');
      }
      const cards = panel.querySelectorAll('.crop-metric');
      if (cards.length !== METRICS.length) throw new Error('대표 지표 표시 영역을 확인하세요.');
      METRICS.forEach((key, i) => {
        const metric = chosen.pooled[key];
        cards[i].querySelector('strong').textContent = format(metric.mean);
        cards[i].querySelector('small').textContent = '전체 평가 행 pooled Test' +
          (count > 1 && metric.std != null ? ` · ${count}시드 평균 · 표준편차 ${format(metric.std)}` : '');
      });
      panel.querySelector('.crop-explanation').textContent = EXPLANATION +
        (count > 1 ? ` 각 시드의 전체 Test 행에서 지표를 계산한 뒤 ${count}시드 지표의 평균과 표본 표준편차를 표시합니다. 시드별 예측값을 먼저 평균한 앙상블 성능은 아닙니다.` : '');
      macroDetails(panel, chosen, count);
      panel.dataset.pooledStatus = 'ready';
    } catch (error) {
      success = false; unavailable(panel, error.message);
    }
  });
  root.dataset.presentationReady = success ? 'ready' : 'error';
  return success;
}
