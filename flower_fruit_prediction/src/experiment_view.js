/**
 * experiment_view.js
 * Renders the Experiment Results Comparison view:
 * - Hierarchical selector controls (Crop -> Target -> Model -> Variable Group -> Split -> Seed)
 * - Variable group detail modal
 * - Candidate combinations comparison table (with sorting & row selection)
 * - Actual vs Predicted time-series chart (ECharts) with sample identity alignment
 * - Line breaks across different entities
 * - Stored overall metrics vs current filtered sample metrics
 * - Hyperparameters card
 * - Status banners for completed / in-progress / unstarted targets
 */

import { fetchMetadata, fetchComparison, fetchPredictions, refreshExperiments } from './experiment_api.js';

// Crop & Target definition mappings
const CROP_TARGETS = {
  tomato: [
    { id: 'tomato_first', label: '1화방 꽃수', part: 'first' },
    { id: 'tomato_second', label: '2화방 꽃수', part: 'second' },
    { id: 'tomato_third', label: '3화방 꽃수', part: 'third' },
    { id: 'tomato_sum123', label: '1~3화방 합계 꽃수 (실험 합계)', part: 'sum123', notice: '※ 1~3화방 합산이며, 상위 화방을 포함하는 기존 관측의 전체 꽃수와 구분됩니다.' },
  ],
  strawberry: [
    { id: 'strawberry_first', label: '1화방 착과수', part: 'first' },
    { id: 'strawberry_second', label: '2화방 착과수', part: 'second' },
    { id: 'strawberry_third', label: '3화방 착과수', part: 'third' },
    { id: 'strawberry_sum123', label: '1~3화방 합계 착과수', part: 'sum123' },
  ],
};

const MODEL_LABELS = {
  poisson: 'Poisson Regression',
  random_forest: 'Random Forest',
  catboost: 'CatBoost',
  mlp: 'MLP (다층 퍼셉트론)',
  tabm: 'TabM',
  tft: 'TFT (Temporal Fusion Transformer)',
};

export class ExperimentViewController {
  constructor() {
    this.metadata = null;
    this.currentComparison = null;
    this.currentPredictions = null;

    // View state
    this.state = {
      crop: 'tomato',
      target: 'tomato_first',
      model: 'tabm',
      group_id: '',
      split: 'validation',
      seed: 'all', // 'all' or specific number
      variant: 'bounded', // 'bounded' or 'raw'
      chartMode: 'individual', // 'individual' or 'average'
      filters: {
        facility_id: 'ALL',
        crop_sn: 'ALL',
        sample_num: 'ALL',
        startDate: '',
        endDate: '',
      },
      sortField: 'rmse',
      sortAsc: true,
    };

    this.chartInstance = null;
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;
    this.initChart();
    this.bindEvents();
    await this.loadInitialData();
    this.initialized = true;
  }

  initChart() {
    const container = document.getElementById('exp_prediction_chart');
    if (!container || !window.echarts) return;
    this.chartInstance = echarts.init(container);
    window.addEventListener('resize', () => {
      if (this.chartInstance) this.chartInstance.resize();
    });
  }

  bindEvents() {
    // 1. Crop toggle
    const cropRadios = document.querySelectorAll('input[name="exp_crop_radio"]');
    cropRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.state.crop = e.target.value;
        const availableTargets = CROP_TARGETS[this.state.crop];
        this.state.target = availableTargets[0].id;
        this.updateTargetDropdown();
        this.onTargetOrModelChange();
      });
    });

    // 2. Target dropdown
    const targetSelect = document.getElementById('exp_target_select');
    if (targetSelect) {
      targetSelect.addEventListener('change', (e) => {
        this.state.target = e.target.value;
        this.onTargetOrModelChange();
      });
    }

    // 3. Model dropdown
    const modelSelect = document.getElementById('exp_model_select');
    if (modelSelect) {
      modelSelect.addEventListener('change', (e) => {
        this.state.model = e.target.value;
        this.onTargetOrModelChange();
      });
    }

    // 4. Variable Group dropdown
    const groupSelect = document.getElementById('exp_group_select');
    if (groupSelect) {
      groupSelect.addEventListener('change', (e) => {
        this.state.group_id = e.target.value;
        this.onGroupChange();
      });
    }

    // 5. Split select (validation / test)
    const splitSelect = document.getElementById('exp_split_select');
    if (splitSelect) {
      splitSelect.addEventListener('change', (e) => {
        this.state.split = e.target.value;
        this.onSplitChange();
      });
    }

    // 6. Seed select
    const seedSelect = document.getElementById('exp_seed_select');
    if (seedSelect) {
      seedSelect.addEventListener('change', (e) => {
        this.state.seed = e.target.value;
        this.loadPredictionsAndRender();
      });
    }

    // 7. Variant toggle (Bounded vs Raw)
    const variantBtns = document.querySelectorAll('.variant-btn');
    variantBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        variantBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.state.variant = btn.dataset.variant;
        this.renderComparisonTable();
        this.renderChartAndMetrics();
      });
    });

    // 8. Chart Mode toggle (individual vs average)
    const modeBtns = document.querySelectorAll('.exp-chart-mode-btn');
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.state.chartMode = btn.dataset.mode;
        this.updatePredictionFilterDropdowns();
        this.renderChartAndMetrics();
      });
    });

    // 9. Prediction Filters
    const facSelect = document.getElementById('exp_filter_facility');
    const cropSelect = document.getElementById('exp_filter_crop_sn');
    const sampleSelect = document.getElementById('exp_filter_sample_num');
    const startInput = document.getElementById('exp_filter_start_date');
    const endInput = document.getElementById('exp_filter_end_date');
    const resetFilterBtn = document.getElementById('exp_filter_reset_btn');

    if (facSelect) {
      facSelect.addEventListener('change', (e) => {
        this.state.filters.facility_id = e.target.value;
        this.state.filters.crop_sn = 'ALL';
        this.state.filters.sample_num = 'ALL';
        this.updatePredictionFilterDropdowns();
        this.renderChartAndMetrics();
      });
    }

    if (cropSelect) {
      cropSelect.addEventListener('change', (e) => {
        this.state.filters.crop_sn = e.target.value;
        this.state.filters.sample_num = 'ALL';
        this.updatePredictionFilterDropdowns();
        this.renderChartAndMetrics();
      });
    }

    if (sampleSelect) {
      sampleSelect.addEventListener('change', (e) => {
        this.state.filters.sample_num = e.target.value;
        this.renderChartAndMetrics();
      });
    }

    if (startInput) {
      startInput.addEventListener('change', (e) => {
        this.state.filters.startDate = e.target.value;
        this.renderChartAndMetrics();
      });
    }

    if (endInput) {
      endInput.addEventListener('change', (e) => {
        this.state.filters.endDate = e.target.value;
        this.renderChartAndMetrics();
      });
    }

    if (resetFilterBtn) {
      resetFilterBtn.addEventListener('click', () => {
        this.state.filters = {
          facility_id: 'ALL',
          crop_sn: 'ALL',
          sample_num: 'ALL',
          startDate: '',
          endDate: '',
        };
        if (startInput) startInput.value = '';
        if (endInput) endInput.value = '';
        this.updatePredictionFilterDropdowns();
        this.renderChartAndMetrics();
      });
    }

    // 10. Refresh button
    const refreshBtn = document.getElementById('exp_refresh_btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        refreshBtn.classList.add('loading');
        try {
          await refreshExperiments();
          await this.loadInitialData();
          this.showToast('실험 데이터가 최신 상태로 새로고침되었습니다.');
        } catch (err) {
          this.showToast(`새로고침 실패: ${err.message}`);
        } finally {
          refreshBtn.classList.remove('loading');
        }
      });
    }

    // 11. Variable details modal
    const detailBtn = document.getElementById('exp_view_features_btn') || document.getElementById('exp_btn_group_detail');
    const modal = document.getElementById('exp_variable_modal');
    const modalCloseBtn = document.getElementById('modal_close_btn') || document.getElementById('exp_modal_close_btn');
    const modalConfirmBtn = document.getElementById('modal_confirm_btn');

    if (detailBtn) {
      detailBtn.addEventListener('click', () => {
        this.openVariableModal();
      });
    }
    if (modalCloseBtn) {
      modalCloseBtn.addEventListener('click', () => {
        if (modal) modal.style.display = 'none';
      });
    }
    if (modalConfirmBtn) {
      modalConfirmBtn.addEventListener('click', () => {
        if (modal) modal.style.display = 'none';
      });
    }
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
      });
    }
  }

  async loadInitialData() {
    try {
      this.metadata = await fetchMetadata();
      this.updateTargetDropdown();
      this.updateModelDropdown();
      await this.onTargetOrModelChange();
    } catch (err) {
      console.error('[ExperimentView] loadInitialData failed:', err);
      this.showToast(`메타데이터 로드 실패: ${err.message}`);
    }
  }

  updateTargetDropdown() {
    const targetSelect = document.getElementById('exp_target_select');
    if (!targetSelect) return;
    targetSelect.innerHTML = '';

    const list = CROP_TARGETS[this.state.crop] || [];
    list.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = item.label;
      if (item.id === this.state.target) opt.selected = true;
      targetSelect.appendChild(opt);
    });

    this.updateTargetNotice();
  }

  updateTargetNotice() {
    const noticeEl = document.getElementById('exp_target_notice');
    if (!noticeEl) return;
    const currentTargetDef = (CROP_TARGETS[this.state.crop] || []).find(t => t.id === this.state.target);
    if (currentTargetDef && currentTargetDef.notice) {
      noticeEl.textContent = currentTargetDef.notice;
      noticeEl.style.display = 'block';
    } else {
      noticeEl.style.display = 'none';
    }
  }

  updateModelDropdown() {
    const modelSelect = document.getElementById('exp_model_select');
    if (!modelSelect) return;
    modelSelect.innerHTML = '';
    const models = this.metadata?.models || Object.keys(MODEL_LABELS);
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = MODEL_LABELS[m] || m;
      if (m === this.state.model) opt.selected = true;
      modelSelect.appendChild(opt);
    });
  }

  async onTargetOrModelChange() {
    this.updateTargetNotice();
    this.renderCampaignStatus();

    try {
      // Fetch comparison table for selected target, model, split
      this.currentComparison = await fetchComparison(this.state.target, this.state.model, this.state.split);
      this.updateGroupDropdown();

      // Automatically select the winner group or first group if empty
      const winner = this.currentComparison.frozen_winner || this.currentComparison.provisional_winner;
      if (winner && this.currentComparison.rows.some(r => r.group_id === winner)) {
        this.state.group_id = winner;
      } else if (this.currentComparison.rows.length > 0) {
        this.state.group_id = this.currentComparison.rows[0].group_id;
      }

      const groupSelect = document.getElementById('exp_group_select');
      if (groupSelect) groupSelect.value = this.state.group_id;

      this.renderComparisonTable();
      await this.loadPredictionsAndRender();
    } catch (err) {
      console.error('[ExperimentView] onTargetOrModelChange error:', err);
      this.showToast(`실험 결과 로드 실패: ${err.message}`);
    }
  }

  updateGroupDropdown() {
    const groupSelect = document.getElementById('exp_group_select');
    if (!groupSelect || !this.currentComparison) return;
    groupSelect.innerHTML = '';

    const rows = this.currentComparison.rows || [];
    const winnerGid = this.currentComparison.frozen_winner;

    rows.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.group_id;
      const badge = (r.group_id === winnerGid) ? '🏆 [최적 확정] ' : (r.rank === 1 ? '⏳ [잠정 1위] ' : '');
      const oldTag = r.old_group_id ? `[${r.old_group_id}] ` : '';
      opt.textContent = `${badge}${oldTag}${r.readable_name} (${r.feature_count}개 변수)`;
      if (r.group_id === this.state.group_id) opt.selected = true;
      groupSelect.appendChild(opt);
    });
  }

  async onGroupChange() {
    // Update selection in comparison table
    this.highlightComparisonTableRow();
    await this.loadPredictionsAndRender();
  }

  async onSplitChange() {
    const testNotice = document.getElementById('exp_split_notice');
    if (testNotice) {
      if (this.state.split === 'test') {
        testNotice.innerHTML = 'ℹ️ <strong>Test 세트 평가 원칙</strong>: 모델별로 최종 선정된 최적 변수군(Frozen Winner)에 대해서만 Test 평가가 수행됩니다. 타 변수군에 Test 결과가 없는 것은 정상입니다.';
        testNotice.style.display = 'block';
      } else {
        testNotice.style.display = 'none';
      }
    }
    await this.onTargetOrModelChange();
  }

  renderCampaignStatus() {
    const banner = document.getElementById('exp_status_banner');
    if (!banner || !this.metadata) return;

    const tMeta = this.metadata.targets?.[this.state.target];
    const status = tMeta?.status || 'unknown';
    const frozenWinner = tMeta?.frozen?.winners?.[this.state.model];
    const overallWinner = tMeta?.frozen?.overall;

    let badgeHtml = '';
    let descHtml = '';

    if (status === 'complete') {
      badgeHtml = '<span class="status-badge complete">✅ 실험 완료 (Frozen Selection)</span>';
      descHtml = `<span>8개 타깃 전체 평가 완료 | <strong>${tMeta.crop_kr} ${tMeta.target_kr}</strong> 전체 최적 모델: <strong>${MODEL_LABELS[overallWinner] || overallWinner}</strong></span>`;
      if (frozenWinner) {
        const rmseVal = (frozenWinner.validation_rmse !== undefined) ? frozenWinner.validation_rmse.toFixed(4) : '-';
        descHtml += ` | <span>현재 모델(${MODEL_LABELS[this.state.model]}) 최적 RMSE: <strong>${rmseVal}</strong></span>`;
      }
    } else if (status === 'in_progress') {
      badgeHtml = '<span class="status-badge in-progress">⚡ 평가 진행 중</span>';
      descHtml = '<span>현재 워커가 해당 타깃의 모델별 튜닝 및 검증을 실시간 실행 중입니다. 잠정 결과가 표시됩니다.</span>';
    } else if (status === 'queued') {
      badgeHtml = '<span class="status-badge queued">⏳ 실행 대기 중</span>';
      descHtml = '<span>스케줄러 큐에 등록되어 선행 타깃 완료 후 순차 실행됩니다.</span>';
    } else {
      badgeHtml = '<span class="status-badge unstarted">⚪ 미실행</span>';
      descHtml = '<span>아직 실행되지 않은 타깃입니다.</span>';
    }

    banner.innerHTML = `<div class="status-banner-content">${badgeHtml} ${descHtml}</div>`;
  }

  renderComparisonTable() {
    const tbody = document.getElementById('exp_comparison_tbody');
    const countBadge = document.getElementById('exp_candidates_count_badge');
    if (!tbody || !this.currentComparison) return;

    const rows = [...(this.currentComparison.rows || [])];
    if (countBadge) {
      countBadge.textContent = `후보 변수군 ${rows.length}개`;
    }

    // Sort rows
    const isAsc = this.state.sortAsc;
    const field = this.state.sortField;
    const variant = this.state.variant; // 'bounded' or 'raw'

    rows.sort((a, b) => {
      let va = null, vb = null;
      if (field === 'rank') {
        va = a.rank; vb = b.rank;
      } else if (field === 'features') {
        va = a.feature_count; vb = b.feature_count;
      } else if (field === 'seeds') {
        va = a.completed_seeds; vb = b.completed_seeds;
      } else if (field === 'rmse') {
        va = a[variant]?.rmse_mean ?? 999999;
        vb = b[variant]?.rmse_mean ?? 999999;
      } else if (field === 'mae') {
        va = a[variant]?.mae_mean ?? 999999;
        vb = b[variant]?.mae_mean ?? 999999;
      } else if (field === 'r2') {
        va = a[variant]?.r2_mean ?? -999999;
        vb = b[variant]?.r2_mean ?? -999999;
      } else if (field === 'ccc') {
        va = a[variant]?.ccc_mean ?? -999999;
        vb = b[variant]?.ccc_mean ?? -999999;
      } else {
        va = a.group_id; vb = b.group_id;
      }
      if (va < vb) return isAsc ? -1 : 1;
      if (va > vb) return isAsc ? 1 : -1;
      return 0;
    });

    tbody.innerHTML = '';
    const winnerGid = this.currentComparison.frozen_winner;

    rows.forEach(r => {
      const tr = document.createElement('tr');
      tr.dataset.groupId = r.group_id;
      if (r.group_id === this.state.group_id) {
        tr.classList.add('selected');
      }
      if (r.group_id === winnerGid) {
        tr.classList.add('winner-row');
      }

      // Winner Badge
      let winnerBadge = '<span class="text-muted">-</span>';
      if (r.group_id === winnerGid) {
        winnerBadge = '<span class="badge-winner">🏆 최적 확정</span>';
      } else if (r.rank === 1 && !winnerGid) {
        winnerBadge = '<span class="badge-provisional">⏳ 잠정 1위</span>';
      }

      // Metrics formatted
      const mData = r[variant] || {};
      const formatMeanStd = (mean, std) => {
        if (mean === undefined || mean === null) return '<span class="text-muted">-</span>';
        if (std && std > 0) {
          return `${mean.toFixed(4)} <small class="text-secondary">±${std.toFixed(4)}</small>`;
        }
        return `${mean.toFixed(4)}`;
      };

      const rmseStr = formatMeanStd(mData.rmse_mean, mData.rmse_std);
      const maeStr = formatMeanStd(mData.mae_mean, mData.mae_std);
      const r2Str = (mData.r2_mean !== undefined && mData.r2_mean !== null) ? mData.r2_mean.toFixed(4) : '<span class="text-muted">-</span>';
      const cccStr = (mData.ccc_mean !== undefined && mData.ccc_mean !== null) ? mData.ccc_mean.toFixed(4) : '<span class="text-muted">-</span>';

      // Status label
      let statusLabel = '<span class="status-tag unstarted">미실행</span>';
      if (r.completed_seeds >= 5) {
        statusLabel = '<span class="status-tag complete">완료 (5/5)</span>';
      } else if (r.completed_seeds > 0) {
        statusLabel = `<span class="status-tag partial">잠정 (${r.completed_seeds}/5)</span>`;
      }

      tr.innerHTML = `
        <td class="td-center font-bold">${r.rank || '-'}</td>
        <td class="td-center">${winnerBadge}</td>
        <td>
          <div class="group-name-cell">
            <strong>${r.readable_name}</strong>
            <small class="text-secondary">${r.old_group_id ? `[${r.old_group_id}] ` : ''}${r.group_id}</small>
          </div>
        </td>
        <td class="td-center font-mono">${r.feature_count}개</td>
        <td class="td-center"><span class="split-pill ${this.state.split}">${this.state.split}</span></td>
        <td class="td-center font-mono">${r.completed_seeds}/5</td>
        <td class="td-right font-mono">${rmseStr}</td>
        <td class="td-right font-mono">${maeStr}</td>
        <td class="td-right font-mono">${r2Str}</td>
        <td class="td-right font-mono">${cccStr}</td>
        <td class="td-center">${statusLabel}</td>
      `;

      // Row click selection
      tr.addEventListener('click', () => {
        this.state.group_id = r.group_id;
        const groupSelect = document.getElementById('exp_group_select');
        if (groupSelect) groupSelect.value = r.group_id;
        this.highlightComparisonTableRow();
        this.loadPredictionsAndRender();
      });

      tbody.appendChild(tr);
    });

    this.bindTableSortHeaders();
  }

  bindTableSortHeaders() {
    const headers = document.querySelectorAll('#exp_comparison_table th[data-sort]');
    headers.forEach(th => {
      th.style.cursor = 'pointer';
      th.onclick = () => {
        const sortKey = th.dataset.sort;
        if (this.state.sortField === sortKey) {
          this.state.sortAsc = !this.state.sortAsc;
        } else {
          this.state.sortField = sortKey;
          this.state.sortAsc = true;
        }
        headers.forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
        th.classList.add(this.state.sortAsc ? 'sorted-asc' : 'sorted-desc');
        this.renderComparisonTable();
      };
    });
  }

  highlightComparisonTableRow() {
    const rows = document.querySelectorAll('#exp_comparison_tbody tr');
    rows.forEach(tr => {
      if (tr.dataset.groupId === this.state.group_id) {
        tr.classList.add('selected');
        tr.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        tr.classList.remove('selected');
      }
    });
  }

  async loadPredictionsAndRender() {
    const chartLoading = document.getElementById('exp_chart_loading');
    if (chartLoading) chartLoading.style.display = 'flex';

    try {
      this.currentPredictions = await fetchPredictions(
        this.state.target,
        this.state.group_id,
        this.state.model,
        this.state.split,
        this.state.seed
      );
      this.updatePredictionFilterDropdowns();
      this.renderChartAndMetrics();
    } catch (err) {
      console.error('[ExperimentView] loadPredictionsAndRender error:', err);
      this.showToast(`예측값 로드 실패: ${err.message}`);
    } finally {
      if (chartLoading) chartLoading.style.display = 'none';
    }
  }

  updatePredictionFilterDropdowns() {
    const facSelect = document.getElementById('exp_filter_facility');
    const cropSelect = document.getElementById('exp_filter_crop_sn');
    const sampleSelect = document.getElementById('exp_filter_sample_num');

    if (!facSelect || !this.currentPredictions?.predictions) return;

    const rows = this.currentPredictions.predictions || [];
    const currentFac = this.state.filters.facility_id;
    const currentCrop = this.state.filters.crop_sn;
    const currentSample = this.state.filters.sample_num;

    // Distinct facilities
    const facilities = Array.from(new Set(rows.map(r => r.facility_id))).filter(Boolean).sort();
    if (currentFac !== 'ALL' && !facilities.includes(currentFac)) {
      this.state.filters.facility_id = 'ALL';
    }
    facSelect.innerHTML = '<option value="ALL">전체 시설</option>';
    facilities.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      if (f === this.state.filters.facility_id) opt.selected = true;
      facSelect.appendChild(opt);
    });

    // Distinct crop SNs (filtered by facility if selected)
    const facRows = (this.state.filters.facility_id === 'ALL') ? rows : rows.filter(r => r.facility_id === this.state.filters.facility_id);
    const cropSns = Array.from(new Set(facRows.map(r => r.crop_sn))).filter(Boolean).sort((a, b) => Number(a) - Number(b));
    if (currentCrop !== 'ALL' && !cropSns.includes(currentCrop)) {
      this.state.filters.crop_sn = 'ALL';
    }
    cropSelect.innerHTML = '<option value="ALL">전체 작기 (기간)</option>';
    cropSns.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = `작기 ${c}`;
      if (c === this.state.filters.crop_sn) opt.selected = true;
      cropSelect.appendChild(opt);
    });

    // Distinct sample nums (filtered by facility & crop if selected)
    const sampleRows = facRows.filter(r => (this.state.filters.crop_sn === 'ALL' || r.crop_sn === this.state.filters.crop_sn));
    const sampleNums = Array.from(new Set(sampleRows.map(r => r.sample_num))).filter(Boolean).sort((a, b) => Number(a) - Number(b));
    if (currentSample !== 'ALL' && !sampleNums.includes(currentSample)) {
      this.state.filters.sample_num = 'ALL';
    }
    sampleSelect.innerHTML = '<option value="ALL">전체 개체</option>';
    sampleNums.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = `#${s} 개체`;
      if (s === this.state.filters.sample_num) opt.selected = true;
      sampleSelect.appendChild(opt);
    });

    // Update status text
    const statusEl = document.getElementById('exp_filter_status');
    if (statusEl) {
      const modeText = this.state.chartMode === 'average' ? '일자별 표본 평균' : '개체별 상세 추이';
      const facText = this.state.filters.facility_id === 'ALL' ? '전체 시설' : this.state.filters.facility_id;
      const cropText = this.state.filters.crop_sn === 'ALL' ? '전체 작기' : `작기 ${this.state.filters.crop_sn}`;
      const sampleText = this.state.filters.sample_num === 'ALL' ? '전체 개체' : `#${this.state.filters.sample_num} 개체`;
      statusEl.innerHTML = `<span>ℹ️ 현재 조건: <strong>${modeText}</strong> (${facText} / ${cropText} / ${sampleText})</span>`;
    }
  }

  filterPredictions() {
    if (!this.currentPredictions?.predictions) return [];
    let rows = this.currentPredictions.predictions;
    const f = this.state.filters;

    if (f.facility_id !== 'ALL') {
      rows = rows.filter(r => r.facility_id === f.facility_id);
    }
    if (f.crop_sn !== 'ALL') {
      rows = rows.filter(r => r.crop_sn === f.crop_sn);
    }
    if (f.sample_num !== 'ALL') {
      rows = rows.filter(r => r.sample_num === f.sample_num);
    }
    if (f.startDate) {
      rows = rows.filter(r => r.target_date >= f.startDate);
    }
    if (f.endDate) {
      rows = rows.filter(r => r.target_date <= f.endDate);
    }
    return rows;
  }

  renderChartAndMetrics() {
    const emptyState = document.getElementById('exp_empty_predictions');
    const chartCard = document.getElementById('exp_chart_card');

    if (!this.currentPredictions || this.currentPredictions.status !== 'success' || !this.currentPredictions.predictions?.length) {
      if (emptyState) {
        emptyState.style.display = 'block';
        const msg = (this.state.split === 'test')
          ? '선택된 변수군은 Test 평가 대상이 아닙니다. (Test 평가는 모델별 확정 최적 변수군에만 존재합니다.)'
          : '선택한 조건에 해당하는 예측 결과가 아직 없거나 진행 중입니다.';
        emptyState.querySelector('.empty-message').textContent = msg;
      }
      if (chartCard) chartCard.style.display = 'none';
      return;
    }

    if (emptyState) emptyState.style.display = 'none';
    if (chartCard) chartCard.style.display = 'block';

    const filteredRows = this.filterPredictions();
    this.renderMetricsCards(filteredRows);
    this.renderEChart(filteredRows);
  }

  renderMetricsCards(filteredRows) {
    const meta = this.currentPredictions.metadata || {};
    const variant = this.state.variant; // 'bounded' or 'raw'
    const stored = meta[variant] || {};

    // 1. Stored overall metrics
    const storedRmseEl = document.getElementById('metric_stored_rmse');
    const storedMaeEl = document.getElementById('metric_stored_mae');
    const storedR2El = document.getElementById('metric_stored_r2');
    const storedCccEl = document.getElementById('metric_stored_ccc');
    const storedNEl = document.getElementById('metric_stored_n');

    const fmt = (v) => (v !== undefined && v !== null && !isNaN(v)) ? Number(v).toFixed(4) : '-';

    if (storedRmseEl) storedRmseEl.textContent = fmt(stored.rmse_mean ?? stored.rmse);
    if (storedMaeEl) storedMaeEl.textContent = fmt(stored.mae_mean ?? stored.mae);
    if (storedR2El) storedR2El.textContent = fmt(stored.r2_mean ?? stored.r2);
    if (storedCccEl) storedCccEl.textContent = fmt(stored.ccc_mean ?? stored.ccc);
    if (storedNEl) storedNEl.textContent = `${meta.n_eval || this.currentPredictions.predictions.length}건`;

    // 2. Current Filtered sample metrics
    const isFiltered = (
      this.state.filters.facility_id !== 'ALL' ||
      this.state.filters.crop_sn !== 'ALL' ||
      this.state.filters.sample_num !== 'ALL' ||
      this.state.filters.startDate ||
      this.state.filters.endDate
    );

    const filteredCard = document.getElementById('exp_filtered_metrics_card');
    const filteredRmseEl = document.getElementById('metric_filtered_rmse');
    const filteredMaeEl = document.getElementById('metric_filtered_mae');
    const filteredR2El = document.getElementById('metric_filtered_r2');
    const filteredNEl = document.getElementById('metric_filtered_n');

    if (filteredRows.length > 0) {
      const predKey = (variant === 'bounded') ? 'prediction_bounded' : 'prediction_raw';
      const actuals = [];
      const preds = [];

      filteredRows.forEach(r => {
        if (r.target !== null && r[predKey] !== null) {
          actuals.push(Number(r.target));
          preds.push(Number(r[predKey]));
        }
      });

      if (actuals.length > 0) {
        // Calculate MSE, RMSE, MAE, R2
        let sumSqErr = 0;
        let sumAbsErr = 0;
        let sumActual = 0;
        for (let i = 0; i < actuals.length; i++) {
          const err = preds[i] - actuals[i];
          sumSqErr += err * err;
          sumAbsErr += Math.abs(err);
          sumActual += actuals[i];
        }
        const meanActual = sumActual / actuals.length;
        let ssTot = 0;
        for (let i = 0; i < actuals.length; i++) {
          const diff = actuals[i] - meanActual;
          ssTot += diff * diff;
        }

        const rmse = Math.sqrt(sumSqErr / actuals.length);
        const mae = sumAbsErr / actuals.length;
        const r2 = (ssTot > 1e-9) ? 1 - (sumSqErr / ssTot) : null;

        if (filteredRmseEl) filteredRmseEl.textContent = rmse.toFixed(4);
        if (filteredMaeEl) filteredMaeEl.textContent = mae.toFixed(4);
        if (filteredR2El) filteredR2El.textContent = (r2 !== null) ? r2.toFixed(4) : '-';
        if (filteredNEl) filteredNEl.textContent = `${actuals.length}건`;
      }
    }

    if (filteredCard) {
      filteredCard.style.display = isFiltered ? 'block' : 'none';
    }

    // 3. Hyperparameters & Meta Card
    const paramsContainer = document.getElementById('exp_hyperparameters_tags');
    if (paramsContainer) {
      const params = meta.params || {};
      paramsContainer.innerHTML = '';

      // Add key params tags
      const addTag = (k, v) => {
        const tag = document.createElement('span');
        tag.className = 'param-tag';
        tag.innerHTML = `<strong>${k}:</strong> ${v}`;
        paramsContainer.appendChild(tag);
      };

      Object.entries(params).forEach(([k, v]) => {
        addTag(k, JSON.stringify(v));
      });

      if (meta.best_epoch) addTag('best_epoch', meta.best_epoch);
      if (meta.upper_bound !== undefined) addTag('upper_bound', meta.upper_bound);
      if (meta.output_clip_rate !== undefined) addTag('output_clip_rate', `${(meta.output_clip_rate * 100).toFixed(1)}%`);
      if (this.currentPredictions.mode === 'ensemble') {
        addTag('앙상블 seed 수', `${this.currentPredictions.seeds_count}개`);
      }
    }
  }

  renderEChart(filteredRows) {
    if (!this.chartInstance) return;

    const variant = this.state.variant; // 'bounded' or 'raw'
    const predKey = (variant === 'bounded') ? 'prediction_bounded' : 'prediction_raw';
    const mode = this.state.chartMode; // 'individual' or 'average'

    const targetDef = (CROP_TARGETS[this.state.crop] || []).find(t => t.id === this.state.target);
    const metricTitle = targetDef?.label || '관측값/예측값';

    let option = {};

    if (mode === 'average') {
      // Aggregate by target_date across all evaluation samples on that day
      const dateMap = new Map();
      filteredRows.forEach(r => {
        if (!r.target_date) return;
        const d = r.target_date;
        if (!dateMap.has(d)) {
          dateMap.set(d, { actuals: [], preds: [], n: 0 });
        }
        const item = dateMap.get(d);
        if (r.target !== null && r[predKey] !== null) {
          item.actuals.push(Number(r.target));
          item.preds.push(Number(r[predKey]));
          item.n += 1;
        }
      });

      const sortedDates = Array.from(dateMap.keys()).sort();
      const actualSeriesData = [];
      const predSeriesData = [];

      sortedDates.forEach(d => {
        const item = dateMap.get(d);
        if (item.actuals.length > 0) {
          const avgActual = item.actuals.reduce((a, b) => a + b, 0) / item.actuals.length;
          const avgPred = item.preds.reduce((a, b) => a + b, 0) / item.preds.length;
          actualSeriesData.push([d, Number(avgActual.toFixed(3)), item.n]);
          predSeriesData.push([d, Number(avgPred.toFixed(3)), item.n]);
        }
      });

      option = {
        title: {
          text: `[일자별 평가 표본 평균] ${metricTitle} - 실측 vs 예측 (${variant.toUpperCase()})`,
          left: 10,
          top: 10,
          textStyle: { fontSize: 14, fontWeight: 'bold', color: '#1e293b' },
          subtext: `동일 일자에 조사된 유효 평가 표본(${filteredRows.length}건)의 평균 집계 (식별키 1:1 매칭)`,
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          textStyle: { color: '#0f172a' },
          formatter: (params) => {
            if (!params || !params.length) return '';
            const dateStr = params[0].axisValueLabel || params[0].value[0];
            const sampleCount = params[0].value[2] || 1;
            let html = `<div style="font-weight:bold;margin-bottom:4px;border-bottom:1px solid #e2e8f0;padding-bottom:2px;">📅 조사일: ${dateStr} (N=${sampleCount}개체)</div>`;
            let actualVal = null;
            let predVal = null;
            params.forEach(p => {
              const val = p.value[1];
              if (p.seriesName.includes('실제')) actualVal = val;
              if (p.seriesName.includes('예측')) predVal = val;
              html += `<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${p.color};"></span>
                <span>${p.seriesName}: <strong>${val}</strong></span>
              </div>`;
            });
            if (actualVal !== null && predVal !== null) {
              const diff = (predVal - actualVal).toFixed(3);
              const sign = diff >= 0 ? `+${diff}` : `${diff}`;
              html += `<div style="margin-top:4px;padding-top:4px;border-top:1px dashed #cbd5e1;color:#64748b;font-size:12px;">오차 (예측 - 실측): <strong>${sign}</strong></div>`;
            }
            return html;
          },
        },
        legend: {
          top: 10,
          right: 20,
          data: ['실제 관측값 평균 (Actual)', '모델 예측값 평균 (Predicted)'],
        },
        grid: { left: 55, right: 30, top: 75, bottom: 65 },
        xAxis: {
          type: 'time',
          boundaryGap: false,
          axisLine: { lineStyle: { color: '#94a3b8' } },
        },
        yAxis: {
          type: 'value',
          name: '개체당 평균 (개)',
          axisLine: { lineStyle: { color: '#94a3b8' } },
          splitLine: { lineStyle: { color: '#f1f5f9' } },
        },
        dataZoom: [
          { type: 'slider', bottom: 10, height: 22, borderColor: '#cbd5e1' },
          { type: 'inside' },
        ],
        series: [
          {
            name: '실제 관측값 평균 (Actual)',
            type: 'line',
            data: actualSeriesData,
            smooth: false,
            symbol: 'circle',
            symbolSize: 6,
            itemStyle: { color: '#10b981' },
            lineStyle: { width: 2.5, color: '#10b981' },
          },
          {
            name: '모델 예측값 평균 (Predicted)',
            type: 'line',
            data: predSeriesData,
            smooth: false,
            symbol: 'diamond',
            symbolSize: 7,
            itemStyle: { color: '#8b5cf6' },
            lineStyle: { width: 2.5, type: 'dashed', color: '#8b5cf6' },
          },
        ],
      };

    } else {
      // Individual Entity Mode: Aligned by (facility_id, crop_sn, sample_num, target_date)
      // Line break when entity changes!
      // Group by entity key: `${facility_id}__${crop_sn}__${sample_num}`
      const entityGroups = new Map();

      filteredRows.forEach(r => {
        if (!r.target_date) return;
        const eKey = `${r.facility_id} | 작기${r.crop_sn} | #${r.sample_num}`;
        if (!entityGroups.has(eKey)) {
          entityGroups.set(eKey, []);
        }
        entityGroups.get(eKey).push(r);
      });

      // Sort each entity's rows by target_date
      entityGroups.forEach(rows => {
        rows.sort((a, b) => (a.target_date < b.target_date ? -1 : 1));
      });

      // Build series data with null separating different entities to prevent line connecting across entities
      const actualData = [];
      const predData = [];

      entityGroups.forEach((rows, eKey) => {
        rows.forEach(r => {
          actualData.push({
            value: [r.target_date, r.target, eKey, r.row_id],
            entity: eKey,
          });
          predData.push({
            value: [r.target_date, r[predKey], eKey, r.row_id],
            entity: eKey,
          });
        });
        // Insert null disconnect to break line between different entities
        actualData.push({ value: [null, null] });
        predData.push({ value: [null, null] });
      });

      option = {
        title: {
          text: `[개체별 상세 추이] ${metricTitle} - 실측 vs 예측 (${variant.toUpperCase()})`,
          left: 10,
          top: 10,
          textStyle: { fontSize: 14, fontWeight: 'bold', color: '#1e293b' },
          subtext: `조회 표본: ${entityGroups.size}개 개체, 총 ${filteredRows.length}개 관측점 (개체간 선 단절 처리)`,
        },
        tooltip: {
          trigger: 'item',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          textStyle: { color: '#0f172a' },
          formatter: (param) => {
            const data = param.data;
            if (!data || !data.value || data.value[0] === null) return '';
            const dateStr = data.value[0];
            const val = data.value[1];
            const entityStr = data.value[2] || '';
            const rowId = data.value[3] || '';

            // Find matching pair row
            const matchRow = filteredRows.find(r => r.row_id === rowId);
            const actualVal = matchRow ? matchRow.target : null;
            const predVal = matchRow ? matchRow[predKey] : null;

            let html = `<div style="font-weight:bold;margin-bottom:4px;border-bottom:1px solid #e2e8f0;padding-bottom:2px;">🏷️ ${entityStr}</div>`;
            html += `<div style="color:#64748b;font-size:12px;margin-bottom:4px;">예측 대상일자: <strong>${dateStr}</strong></div>`;
            html += `<div style="color:#10b981;">실제 관측값: <strong>${actualVal !== null ? actualVal : '-'}</strong></div>`;
            html += `<div style="color:#8b5cf6;">모델 예측값: <strong>${predVal !== null ? predVal.toFixed(3) : '-'}</strong></div>`;

            if (actualVal !== null && predVal !== null) {
              const diff = (predVal - actualVal).toFixed(3);
              const sign = diff >= 0 ? `+${diff}` : `${diff}`;
              html += `<div style="margin-top:4px;padding-top:4px;border-top:1px dashed #cbd5e1;color:#0ea5e9;font-size:12px;">오차: <strong>${sign}</strong></div>`;
            }
            return html;
          },
        },
        legend: {
          top: 10,
          right: 20,
          data: ['실제 관측값 (Actual)', '모델 예측값 (Predicted)'],
        },
        grid: { left: 55, right: 30, top: 75, bottom: 65 },
        xAxis: {
          type: 'time',
          boundaryGap: false,
          axisLine: { lineStyle: { color: '#94a3b8' } },
        },
        yAxis: {
          type: 'value',
          name: '개체 관측값 (개)',
          axisLine: { lineStyle: { color: '#94a3b8' } },
          splitLine: { lineStyle: { color: '#f1f5f9' } },
        },
        dataZoom: [
          { type: 'slider', bottom: 10, height: 22, borderColor: '#cbd5e1' },
          { type: 'inside' },
        ],
        series: [
          {
            name: '실제 관측값 (Actual)',
            type: 'line',
            data: actualData,
            smooth: false,
            connectNulls: false, // DO NOT connect lines when entity changes!
            symbol: 'circle',
            symbolSize: 5,
            itemStyle: { color: '#10b981' },
            lineStyle: { width: 1.8, color: '#10b981' },
          },
          {
            name: '모델 예측값 (Predicted)',
            type: 'line',
            data: predData,
            smooth: false,
            connectNulls: false, // DO NOT connect lines when entity changes!
            symbol: 'diamond',
            symbolSize: 6,
            itemStyle: { color: '#8b5cf6' },
            lineStyle: { width: 1.8, type: 'dashed', color: '#8b5cf6' },
          },
        ],
      };
    }

    this.chartInstance.setOption(option, true);
  }

  openVariableModal() {
    const modal = document.getElementById('exp_variable_modal');
    if (!modal) return;

    if (!this.metadata) {
      this.loadInitialData().then(() => this.openVariableModal());
      return;
    }

    const tData = this.metadata.targets?.[this.state.target];
    let groupInfo = tData?.groups?.[this.state.group_id];

    // Fallback to first group if not selected yet
    if (!groupInfo && tData?.groups) {
      const firstGid = Object.keys(tData.groups)[0];
      if (firstGid) {
        this.state.group_id = firstGid;
        groupInfo = tData.groups[firstGid];
      }
    }

    if (!groupInfo) {
      this.showToast('선택된 변수군 정보를 찾을 수 없습니다.');
      return;
    }

    const titleEl = document.getElementById('modal_group_name') || document.getElementById('modal_group_title');
    const readableEl = document.getElementById('modal_group_readable');
    const idEl = document.getElementById('modal_group_id');
    const oldIdEl = document.getElementById('modal_old_group_id');
    const countEl = document.getElementById('modal_group_count') || document.getElementById('modal_feature_count');
    const catEl = document.getElementById('modal_categories_list');
    const featListEl = document.getElementById('modal_features_container');

    if (titleEl) {
      titleEl.textContent = `${groupInfo.readable_name || '입력변수군 상세'}`;
    }
    if (readableEl) {
      readableEl.textContent = `${groupInfo.group_id}${groupInfo.old_group_id ? ` (${groupInfo.old_group_id})` : ''} • 총 ${groupInfo.feature_count}개 변수`;
    }
    if (idEl) {
      idEl.textContent = groupInfo.group_id;
    }
    if (oldIdEl) {
      oldIdEl.textContent = groupInfo.old_group_id || '-';
    }
    if (countEl) {
      countEl.textContent = `${groupInfo.feature_count}개`;
    }

    if (catEl) {
      catEl.innerHTML = '';
      const cats = groupInfo.category_names || [];
      if (cats.length === 0) {
        catEl.textContent = '-';
      } else {
        cats.forEach(c => {
          const tag = document.createElement('span');
          tag.className = 'cat-tag';
          tag.textContent = c;
          catEl.appendChild(tag);
        });
      }
    }

    if (featListEl) {
      featListEl.innerHTML = '';
      const features = groupInfo.features || [];
      if (features.length === 0) {
        featListEl.innerHTML = '<div class="text-muted" style="padding: 10px;">변수 목록 정보가 없습니다.</div>';
      } else {
        features.forEach((f, idx) => {
          const item = document.createElement('div');
          item.className = 'feature-item';
          item.innerHTML = `<span class="feat-idx">${idx + 1}.</span> <span class="feat-name">${f}</span>`;
          featListEl.appendChild(item);
        });
      }
    }

    modal.style.display = 'flex';
  }

  showToast(msg) {
    const toast = document.getElementById('alert_toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = 'flex';
    setTimeout(() => { toast.style.display = 'none'; }, 3500);
  }
}
