import { selectedPooled } from './presentation.js?v=20261027';

const METRICS = ['rmse', 'mae', 'r2', 'ccc'];
export const GROUP_NAMES = { E1: '목표·생식생육 이력', E2: '영양생장', E3: '시설내환경', E4: '외부기상', E5: '근권양액', E6: '시설제어', E7: '시설·작기 맥락' };
const dataURL = path => new URL(`./expanded_data/${path}`, import.meta.url);
const digest = async text => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))), b => b.toString(16).padStart(2, '0')).join('');

export function calculateMetrics(rows, key) {
  if (!rows.length) return {};
  const n = rows.length, meanY = rows.reduce((s, r) => s + r.target, 0) / n, meanP = rows.reduce((s, r) => s + r[key], 0) / n;
  let mse = 0, mae = 0, vy = 0, vp = 0, cov = 0;
  for (const row of rows) {
    mse += (row.target - row[key]) ** 2 / n; mae += Math.abs(row.target - row[key]) / n;
    vy += (row.target - meanY) ** 2 / n; vp += (row[key] - meanP) ** 2 / n;
    cov += (row.target - meanY) * (row[key] - meanP) / n;
  }
  const denom = vy + vp + (meanY - meanP) ** 2;
  return { rmse_mean: Math.sqrt(mse), mae_mean: mae, r2_mean: vy > 0 ? 1 - mse / vy : null, ccc_mean: denom > 0 ? 2 * cov / denom : null };
}

export class ExpandedData {
  constructor() { this.models = new Map(); this.predictions = new Map(); }
  async load() {
    const res = await fetch(new URL('../summary.json', import.meta.url), { cache: 'no-store' });
    if (!res.ok) throw new Error('현재 검증 결과를 불러오지 못했습니다.');
    const text = await res.text(); this.bundle = JSON.parse(text);
    for (const crop of ['tomato', 'strawberry']) selectedPooled(this.bundle, crop);
    this.summaryHash = await digest(text);
    this.index = null; this.matched = false;
    try {
      const result = await fetch(dataURL('manifest.json'), { cache: 'no-store' });
      if (result.ok) this.index = await result.json();
      this.matched = this.index?.schema === 1 && this.index.summary_sha256 === this.summaryHash;
    } catch { /* Summary Test metrics remain usable; never substitute stale detailed results. */ }
    this.models.clear(); this.predictions.clear();
  }
  target(crop, value) {
    const match = new RegExp(`^${crop}_truss(\\d+)$`).exec(value);
    const truss = match ? Number(match[1]) : NaN;
    if (!Number.isInteger(truss) || truss < 1 || truss > this.bundle.crops[crop].limit) throw new Error('유효하지 않은 화방입니다.');
    return truss;
  }
  async model(crop, model) {
    const key = `${crop}/${model}`;
    if (!this.models.has(key)) this.models.set(key, this.loadModel(crop, model));
    return this.models.get(key);
  }
  async loadModel(crop, model) {
    if (this.matched) {
      const spec = this.index.models[crop][model];
      const response = await fetch(dataURL(spec.path), { cache: 'no-store' });
      if (!response.ok) throw new Error('화방별 조합 결과를 불러오지 못했습니다.');
      const text = await response.text();
      if (await digest(text) !== spec.sha256) throw new Error('화방별 결과 버전이 일치하지 않습니다. 새로고침해 주세요.');
      return JSON.parse(text);
    }
    // A new automatic publication may precede its detailed export. Use only its own Test summary.
    const chosen = this.bundle.crops[crop].models.find(m => m.model === model);
    const combinations = {};
    for (const row of this.bundle.comparison.filter(r => r.crop === crop && r.model === model)) {
      const item = { groups: row.groups.split('+'), features: row.features, selected: [], params: null, validation: null, test: null };
      if (row.groups === chosen.groups) {
        const trusses = {};
        for (const truss of chosen.per_truss) {
          const bounded = {};
          for (const k of METRICS) { bounded[`${k}_mean`] = truss.metrics[k].mean; bounded[`${k}_std`] = truss.metrics[k].std; }
          trusses[truss.truss] = { bounded, n_eval: truss.n, positive_n: truss.positive_n };
        }
        item.test = { trusses, predictions: {} };
      }
      combinations[row.groups] = item;
    }
    return { winner: chosen.groups, combinations };
  }
  async catalog(crop, model) {
    const records = await this.model(crop, model);
    const definitions = this.index?.groups?.[crop] || {};
    const feature = id => ({ feature_id: id, feature_name: id, desc: '화방 확장 실험 파생 입력', type: '모델 입력' });
    const variable_groups = Object.fromEntries(Object.entries(GROUP_NAMES).map(([code, name]) => [code, { code, name, features: (definitions[code] || []).map(feature) }]));
    variable_groups.C0 = { code: 'C0', name: '공통 입력', features: [{ feature_id: 'truss', feature_name: 'truss', desc: '예측 대상 화방 번호 · 모든 조합 공통 입력', type: '정수' }] };
    const combinations = Object.entries(records.combinations).map(([id, r]) => ({
      combination_id: id, groups: r.groups, group_count: r.groups.length, label: r.groups.join('+'),
      group_names: r.groups.map(g => GROUP_NAMES[g]), feature_count: r.features,
      features: r.selected.map(feature), display_name: r.groups.join('+')
    })).sort((a, b) => a.group_count - b.group_count || a.label.localeCompare(b.label));
    return { combinations, variable_groups };
  }
  async comparison(crop, truss, model, split) {
    const data = await this.model(crop, model), catalog = await this.catalog(crop, model);
    const rows = catalog.combinations.map(combo => {
      const r = data.combinations[combo.combination_id], detail = r[split]?.trusses?.[truss];
      return { group_id: combo.combination_id, readable_name: combo.label, group_count: combo.group_count,
        category_ids: combo.groups, feature_count: r.features, params: r.params, best_epoch: r.best_epoch,
        bounded: detail?.bounded || {}, raw: detail?.raw || {}, n_eval: detail?.n_eval, positive_n: detail?.positive_n,
        completed_seeds: detail ? this.bundle.seed_count : 0, is_winner: combo.combination_id === data.winner,
        status: detail ? 'completed' : split === 'test' && combo.combination_id !== data.winner ? 'not_evaluated' : 'detail_unavailable' };
    });
    const ordered = rows.filter(r => Number.isFinite(r.bounded.rmse_mean)).sort((a, b) => a.bounded.rmse_mean - b.bounded.rmse_mean);
    ordered.forEach((row, i) => { row.rank = i + 1; }); // Diagnostic order only; frozen winner is never changed.
    return { rows, frozen_winner: data.winner, total_candidates: rows.length };
  }
  async prediction(crop, truss, model, group, split, seed) {
    const data = await this.model(crop, model), item = data.combinations[group];
    const detail = item?.[split]?.trusses?.[truss];
    const file = item?.[split]?.predictions?.[String(seed)];
    const path = file?.path;
    const metadata = { ...(detail || {}), params: item?.params, prediction_note: this.matched
      ? '개별 예측 시계열은 모델별 Validation 선정 조합의 Validation·Test에 제공합니다. 나머지 Validation 조합은 화방별 평가지표로 확인할 수 있습니다.'
      : '최신 결과의 화방별 상세 예측 파일이 아직 게시되지 않았습니다. 이전 결과를 섞지 않고 현재 summary의 Test 지표만 표시합니다.' };
    if (!path) return { predictions: [], metadata, status: 'no_predictions' };
    if (!this.predictions.has(path)) this.predictions.set(path, (async () => {
      const res = await fetch(dataURL(path), { cache: 'no-store' });
      if (!res.ok) throw new Error('저장된 예측값을 불러오지 못했습니다.');
      const text = await res.text();
      if (await digest(text) !== file.sha256) throw new Error('예측 파일 버전이 일치하지 않습니다. 새로고침해 주세요.');
      const packet = JSON.parse(text);
      if (packet.crop !== crop || packet.model !== model || packet.split !== split || packet.combo !== group || packet.seed !== Number(seed)) throw new Error('예측값의 실험 조건이 일치하지 않습니다.');
      return packet.rows.map(row => Object.fromEntries(packet.columns.map((key, i) => [key, row[i]])));
    })());
    const predictions = (await this.predictions.get(path)).filter(row => row.truss === truss);
    return { predictions, status: 'success', metadata: { ...metadata, n_eval: predictions.length,
      bounded: calculateMetrics(predictions, 'prediction_bounded'), raw: calculateMetrics(predictions, 'prediction_raw') } };
  }
}
