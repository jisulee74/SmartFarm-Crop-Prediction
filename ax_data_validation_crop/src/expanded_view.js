// Reuse the existing selector, table, filters and modals at their original DOM locations.
import { ExperimentViewController as OriginalView } from '../../ax_data_validation/src/experiment_view.js?v=11';
import { ExpandedData, GROUP_NAMES, calculateMetrics } from './expanded_api.js?v=20261102b';

const fmt = value => Number.isFinite(value) ? value.toFixed(3) : '산출 불가';
const LABELS = { tomato: '꽃수 (개화수)', strawberry: '착과수' };

export class ExperimentViewController extends OriginalView {
  constructor() {
    super(); this.data = new ExpandedData(); this.state.campaign = 'upper_truss_full';
    this.state.target = 'tomato_truss1'; this.state.chartMode = 'average';
  }
  key() { return [this.state.crop, this.state.target, this.state.model, this.state.split].join('/'); }
  resetFilters() {
    this.state.filters = { facility_id: 'ALL', crop_sn: 'ALL', sample_num: 'ALL', startDate: '', endDate: '' };
    for (const id of ['exp_filter_start_date', 'exp_filter_end_date']) {
      const field = document.getElementById(id); if (field) field.value = '';
    }
  }
  bindEvents() {
    super.bindEvents();
    // Replace only the old refresh handler; all the original controls keep their existing IDs.
    const old = document.getElementById('exp_refresh_btn');
    const refresh = old.cloneNode(true); old.replaceWith(refresh);
    refresh.addEventListener('click', async () => {
      refresh.disabled = true;
      try { await this.loadInitialData(); } catch (error) { this.showToast(error.message); }
      finally { refresh.disabled = false; }
    });
    document.querySelectorAll('.exp-chart-mode-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === this.state.chartMode));
    document.querySelector('#crop_advanced')?.addEventListener('toggle', () => this.chartInstance?.resize());
    for (const [id, key] of [['exp_filter_start_date','startDate'], ['exp_filter_end_date','endDate']]) {
      document.getElementById(id).addEventListener('change', event => {
        this.state.filters[key] = event.target.value; this.renderPredictionChart();
      });
    }
    document.getElementById('exp_filter_reset_btn').addEventListener('click', () => {
      this.resetFilters(); this.renderPredictionChart();
    });
  }
  async loadInitialData() {
    await this.data.load();
    this.state.crop = document.querySelector('input[name="exp_crop_radio"]:checked')?.value || this.state.crop;
    this.metadata = { seeds: this.data.bundle.seeds, campaign: { total_seeds: this.data.bundle.seed_count, test_seeds: this.data.bundle.seeds } };
    this.state.model = this.data.bundle.crops[this.state.crop].selected;
    document.getElementById('exp_model_select').value = this.state.model;
    this.renderCampaignSelector(); this.updateSeedDropdown(true); this.updateTargetDropdown();
    await this.loadComparisonData(); await this.loadPredictionsAndRender();
    document.querySelector('.exp-control-card').dataset.expandedReady = 'true';
  }
  renderCampaignSelector() {
    const count = this.data.bundle.seed_count;
    document.getElementById('exp_method_summary').textContent =
      `화방 확장 실험 · 토마토 1~20화방 / 딸기 1~5화방 · 127개 조합 × 6개 모델 · ${count}시드. 선정은 작물 전체 Validation pooled RMSE 기준이며, 아래 표는 선택 화방의 진단 지표입니다.`;
    const advanced = document.querySelector('#crop_advanced');
    advanced.open = true;
    advanced.querySelector(':scope > summary').textContent = '화방별 모델·입력 조합 상세 검증 · 토마토 1~20화방 / 딸기 1~5화방';
    document.querySelector('label[for="exp_target_select"]').textContent = '2. 예측 대상 (Target)';
    document.querySelector('[data-variant="bounded"]').textContent = '음수만 0으로 보정';
    document.querySelector('.variant-toggle-wrap .meta-label').textContent = '예측값 표시:';
    document.querySelector('#exp_comparison_table th[data-sort="rank"]').textContent = '화방 내 순위';
    document.querySelector('#exp_comparison_table').closest('section').querySelector('.exp-card-subtitle').textContent =
      '선택 화방의 Validation 지표를 127개 조합별로 비교합니다. Test는 모델별 Validation 선정 조합만 평가했습니다. 화방 내 순위는 진단용이며 작물 전체 기준의 동결된 선정 결과를 변경하지 않습니다.';
    const stored = document.getElementById('metric_stored_rmse').closest('.metric-card');
    const heading = stored.querySelector('.metric-title');
    if (heading) heading.textContent = '선택 화방의 저장된 평가 지표';
    stored.querySelector('.metric-subtitle').textContent = '시계열 제공 시 선택 시드 지표 · 미제공 시 저장된 시드 평균';
    document.querySelector('.metric-card.params .metric-title').textContent = '저장된 학습 설정';
    document.querySelector('.metric-card.params .metric-subtitle').textContent = this.data.bundle.mode === 'quick' ? '1시드 실험의 고정 설정값' : '정식 실험에서 동결한 설정값';
    const note = document.createElement('p'); note.id = 'expanded_scope_note'; note.className = 'crop-explanation';
    note.textContent = `비교표는 ${count}시드 지표 평균입니다. 시계열과 아래 저장 지표는 선택한 단일 시드의 같은 화방·평가 사례를 사용합니다. 원본 시설·작기·개체 식별자는 화면용 번호로 치환했습니다. ` +
      (this.data.matched ? '선정 조합의 Validation·Test 시계열을 제공합니다.' : '현재 요약 결과와 상세 파일의 버전이 달라 최신 Test 요약 지표만 표시합니다. 과거 예측값을 섞지 않습니다.');
    document.getElementById('expanded_scope_note')?.remove();
    document.querySelector('.exp-control-card').append(note);
    for (const [g, name] of Object.entries(GROUP_NAMES)) {
      const button = document.querySelector(`#combo_included_grp_pills [data-grp="${g}"]`);
      if (button) button.textContent = `${g}: ${name}`;
    }
  }
  updateSeedDropdown(reset = false) {
    const select = document.getElementById('exp_seed_select');
    const seeds = this.data.bundle?.seeds || [];
    select.replaceChildren(...seeds.map(seed => new Option(`Seed ${seed}`, String(seed))));
    if (reset || !seeds.includes(Number(this.state.seed))) this.state.seed = String(seeds[0]);
    select.value = this.state.seed;
  }
  updateTargetDropdown() {
    const crop = this.state.crop, limit = this.data.bundle?.crops[crop].limit || (crop === 'tomato' ? 20 : 5);
    const select = document.getElementById('exp_target_select');
    select.replaceChildren(...Array.from({ length: limit }, (_, i) => new Option(`${i+1}화방 ${LABELS[crop]}`, `${crop}_truss${i+1}`)));
    if (![...select.options].some(option => option.value === this.state.target)) this.state.target = `${crop}_truss1`;
    select.value = this.state.target;
    if (this.lastCrop !== crop) {
      this.lastCrop = crop; this.resetFilters();
      this.state.model = this.data.bundle?.crops[crop].selected || this.state.model;
      document.getElementById('exp_model_select').value = this.state.model;
      this.state.group_id = '';
      document.querySelectorAll('[data-summary-crop]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.summaryCrop === crop)));
      document.querySelectorAll('[data-summary-panel]').forEach(panel => { panel.hidden = panel.dataset.summaryPanel !== crop; });
    }
  }
  async loadComparisonData() {
    const key = this.key(), { crop, target, model, split } = this.state;
    const truss = this.data.target(crop, target);
    const [catalog, comparison] = await Promise.all([this.data.catalog(crop, model), this.data.comparison(crop, truss, model, split)]);
    if (key !== this.key()) return;
    this.catalog = catalog; this.currentComparison = comparison;
    if (!catalog.combinations.some(c => c.combination_id === this.state.group_id) || split === 'test') this.state.group_id = comparison.frozen_winner;
    this.populateCombinations(); this.renderComparisonTable(); this.updateGroupMetaBar();
    this.renderDiagnosis();
  }
  async onTargetOrModelChange() {
    this.resetFilters();
    await this.loadComparisonData(); await this.loadPredictionsAndRender();
  }
  async onSplitChange() {
    this.updateSeedDropdown(true);
    // Re-population respects active combination filters; never select an invisible option.
    await this.loadComparisonData(); await this.loadPredictionsAndRender();
  }
  applyCombinationFilters() {
    const before = this.state.group_id;
    super.applyCombinationFilters();
    if (before !== this.state.group_id) this.loadPredictionsAndRender();
  }
  renderComparisonTable() {
    const ranked = (this.currentComparison?.rows || []).filter(r => Number.isFinite(r[this.state.variant]?.rmse_mean))
      .sort((a,b) => a[this.state.variant].rmse_mean - b[this.state.variant].rmse_mean);
    for (const row of this.currentComparison?.rows || []) row.rank = null;
    ranked.forEach((row, i) => { row.rank = i + 1; });
    super.renderComparisonTable();
    const rows = this.currentComparison?.rows || [];
    document.querySelectorAll('#exp_comparison_tbody tr[data-group-id]').forEach(tr => {
      const row = rows.find(r => r.group_id === tr.dataset.groupId);
      const badge = tr.querySelector('.badge-winner');
      if (badge) badge.textContent = 'Validation 선정';
      if (row?.status === 'detail_unavailable') tr.lastElementChild.textContent = '상세 지표 미게시';
      else if (this.state.variant === 'raw' && row?.bounded?.rmse_mean != null && row?.raw?.rmse_mean == null) tr.lastElementChild.textContent = 'Raw 지표 미게시';
    });
  }
  async loadPredictionsAndRender() {
    const key = [this.key(), this.state.group_id, this.state.seed].join('/');
    const { crop, target, model, group_id, split, seed } = this.state;
    const loading = document.getElementById('exp_chart_loading'); loading.style.display = 'flex';
    this.chartInstance?.clear();
    try {
      const result = await this.data.prediction(crop, this.data.target(crop, target), model, group_id, split, seed);
      if (key !== [this.key(), this.state.group_id, this.state.seed].join('/')) return;
      this.currentPredictions = result;
      const has = result.predictions.length > 0;
      document.getElementById('exp_chart_card').style.display = has ? 'block' : 'none';
      document.getElementById('exp_empty_predictions').style.display = has ? 'none' : 'block';
      document.getElementById('exp_filtered_metrics_card').style.display = has ? 'block' : 'none';
      if (has) { this.populateChartFilterDropdowns(); this.renderPredictionChart(); this.renderStoredMetrics(); }
      else this.renderEmptyMetrics();
      document.querySelector('.exp-control-card').dataset.loadedSelection = key;
    } catch (error) {
      if (key !== [this.key(), this.state.group_id, this.state.seed].join('/')) return;
      this.currentPredictions = { predictions: [] };
      document.getElementById('exp_chart_card').style.display = 'none';
      document.getElementById('exp_empty_predictions').style.display = 'block';
      this.renderEmptyMetrics();
      document.getElementById('exp_empty_predictions_msg').textContent = error.message;
    } finally {
      if (key === [this.key(), this.state.group_id, this.state.seed].join('/')) loading.style.display = 'none';
    }
  }
  renderStoredMetrics() {
    super.renderStoredMetrics();
    const el = document.getElementById('metric_stored_ccc');
    if (el) el.textContent = fmt(this.currentPredictions?.metadata?.[this.state.variant]?.ccc_mean);
  }
  renderEmptyMetrics() {
    super.renderEmptyMetrics();
    const row = this.currentComparison?.rows.find(r => r.group_id === this.state.group_id);
    document.getElementById('metric_stored_ccc').textContent = fmt(row?.[this.state.variant]?.ccc_mean);
    if (!this.data.matched) document.getElementById('exp_empty_predictions_msg').textContent =
      '최신 결과의 상세 파일이 아직 게시되지 않았습니다. 현재 결과에 있는 화방별 Test 요약 지표만 표시하며, 이전 실험의 예측값은 사용하지 않습니다.';
  }
  renderPredictionChart() {
    if (!this.currentPredictions?.predictions?.length || !this.chartInstance) return;
    this.renderStoredMetrics();
    const filters = this.state.filters;
    const rows = this.currentPredictions.predictions.filter(row =>
      ['facility_id', 'crop_sn', 'sample_num'].every(k => filters[k] === 'ALL' || String(row[k]) === filters[k]) &&
      (!filters.startDate || row.target_date >= filters.startDate) && (!filters.endDate || row.target_date <= filters.endDate));
    const valueKey = this.state.variant === 'raw' ? 'prediction_raw' : 'prediction_bounded';
    const values = calculateMetrics(rows, valueKey);
    for (const k of ['rmse', 'mae', 'r2', 'ccc']) {
      const node = document.getElementById(`metric_filtered_${k}`);
      if (node) node.textContent = fmt(values[`${k}_mean`]);
    }
    document.getElementById('metric_filtered_n').textContent = `${rows.length}건`;
    document.getElementById('exp_filtered_metrics_card').style.display = 'block';
    const individual = this.state.chartMode === 'individual';
    const onePlant = ['facility_id', 'crop_sn', 'sample_num'].every(k => this.state.filters[k] !== 'ALL');
    let points;
    if (individual) points = rows.map(row => ({ date: row.target_date, actual: row.target, predicted: row[valueKey], label: `${row.facility_id} / ${row.crop_sn} / 개체 ${row.sample_num}` }));
    else {
      const grouped = new Map();
      for (const row of rows) {
        if (!grouped.has(row.target_date)) grouped.set(row.target_date, []);
        grouped.get(row.target_date).push(row);
      }
      points = Array.from(grouped, ([date, values]) => ({ date, actual: values.reduce((s,r) => s+r.target,0)/values.length,
        predicted: values.reduce((s,r) => s+r[valueKey],0)/values.length, label: `같은 평가 사례 ${values.length}건의 평균` }));
    }
    points.sort((a,b) => a.date.localeCompare(b.date));
    this.chartInstance.setOption({
      title: { text: `${document.getElementById('exp_target_select').selectedOptions[0].textContent} · ${individual ? '개체별 관측·예측' : '일자별 표본 평균'}`,
        subtext: `${this.state.split} · ${this.state.model.toUpperCase()} · Seed ${this.state.seed}${individual && !onePlant ? ' · 서로 다른 개체는 점으로 표시' : ''}`,
        textStyle: { fontSize: 14 }, left: 10 },
      tooltip: { trigger: 'item', formatter: p => `${this.escapeHtml(p.data.label)}<br>${p.data.date} · ${p.seriesName}: ${fmt(p.value[1])}` },
      legend: { data: ['실제 관측값', '모델 예측값'], top: 48 },
      grid: { left: 50, right: 25, top: 90, bottom: 75 },
      xAxis: { type: 'time' }, yAxis: { type: 'value', name: '개수' },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 10 }],
      series: [['actual','실제 관측값','#10b981'], ['predicted','모델 예측값','#2563eb']].map(([key,name,color]) => ({
        name, type: individual && !onePlant ? 'scatter' : 'line', smooth: false, symbolSize: 6, itemStyle: { color },
        data: points.map(p => ({ value: [p.date+'T00:00:00Z', p[key]], label: p.label, date: p.date }))
      }))
    }, true);
  }
  openVariableModal() { return this.openFeatureSelectionModal(this.state.group_id); }
  async openFeatureSelectionModal(groupId) {
    const modal = document.getElementById('exp_selection_modal');
    modal.style.display = 'flex';
    const content = document.getElementById('selection_modal_content'); content.textContent = '선택 변수 기록을 불러오는 중입니다.';
    const data = await this.data.model(this.state.crop, this.state.model), item = data.combinations[groupId];
    document.getElementById('selection_modal_title').textContent = `${item.groups.join('+')} · 실제 사용 변수`;
    document.getElementById('selection_modal_subtitle').textContent = '작물별 공통 모델의 동결된 실제 입력 목록 · 화방 번호는 모든 조합의 공통 입력';
    content.replaceChildren();
    const note = document.createElement('p');
    note.textContent = item.selected.length ? `최종 입력 ${item.selected.length}개. 학습·검증 때 저장한 변수 목록입니다. 화방별로 재선정하지 않습니다.` : '현재 결과에 대응하는 실제 선택 변수 상세 파일이 아직 게시되지 않았습니다.';
    content.append(note);
    const table = document.createElement('table'); table.className = 'variable-table';
    const head = table.createTHead().insertRow();
    for (const name of ['변수군', '실제 입력 변수 ID']) { const th = document.createElement('th'); th.textContent = name; head.append(th); }
    const body = table.createTBody();
    for (const name of item.selected) {
      const row = body.insertRow();
      const group = name === 'truss' ? '공통 화방 번호' : Object.entries(this.data.index.groups[this.state.crop]).find(([, names]) => names.includes(name))?.[0] || '확인 필요';
      row.insertCell().textContent = group; row.insertCell().textContent = name;
    }
    content.append(table);
  }
  renderDiagnosis() {
    const pane = document.getElementById('model-subtab-diagnosis');
    if (!pane || !this.data.bundle) return;
    for (const child of pane.children) if (child.id !== 'expanded_diagnosis') child.hidden = true;
    let section = document.getElementById('expanded_diagnosis');
    if (!section) { section = document.createElement('section'); section.id = 'expanded_diagnosis'; section.className = 'exp-section-card'; pane.append(section); }
    section.replaceChildren();
    const spec = this.data.bundle.crops[this.state.crop], chosen = spec.models.find(m => m.model === spec.selected);
    const title = document.createElement('h3'); title.textContent = `${this.state.crop === 'tomato' ? '토마토' : '딸기'} · 대표 모델의 화방별 Test 오차`;
    const note = document.createElement('p'); note.textContent = `${chosen.model.toUpperCase()} / ${chosen.groups} · ${this.data.bundle.seed_count}시드 지표 평균. Validation 선정 결과를 유지하며 현재값 유지 기준선과 비교합니다.`;
    const wrap = document.createElement('div'); wrap.className = 'crop-table-scroll';
    const table = document.createElement('table'), head = table.createTHead().insertRow(), body = table.createTBody();
    for (const name of ['화방', '평가 건수', '양수 건수', 'RMSE', 'MAE', 'R²', 'CCC', '현재값 유지 RMSE']) { const th = document.createElement('th'); th.textContent = name; head.append(th); }
    for (const r of chosen.per_truss) {
      const baseline = spec.baselines.persistence.per_truss.find(b => b.truss === r.truss);
      const row = body.insertRow();
      for (const value of [r.truss, r.n, r.positive_n, ...['rmse', 'mae', 'r2', 'ccc'].map(k => fmt(r.metrics[k].mean)), fmt(baseline?.rmse)]) row.insertCell().textContent = value;
    }
    wrap.append(table); section.append(title, note, wrap);
  }
}
