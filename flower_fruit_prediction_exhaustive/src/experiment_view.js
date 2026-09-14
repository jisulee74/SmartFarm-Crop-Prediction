/**
 * experiment_view.js
 * Renders the 127 Variable Group Combinations Exhaustive Experiment View:
 * - Hierarchical selector controls (Crop -> Target -> Model -> 127 Combinations -> Split -> Seed)
 * - Multi-dimensional Combination Filter Toolbar (Group Count 1~7, Included Groups E1~E7, Live Search)
 * - 127 Combinations Explorer Modal with responsive grid cards
 * - Variable Details Modal with comprehensive 49-feature dictionary & fixed C0 metadata
 * - Candidate combinations comparison table (with robust sorting & pending status handling)
 * - Actual vs Predicted time-series chart (ECharts) with sample identity alignment & cascading dropdowns
 * - Stored overall metrics vs current filtered sample metrics
 * - Hyperparameters card
 */

import { fetchCatalog, fetchMetadata, fetchComparison, fetchPredictions, refreshExperiments } from './experiment_api.js';

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
  poisson: 'Poisson Regression (CPU)',
  random_forest: 'Random Forest (CPU)',
  catboost: 'CatBoost (GPU)',
  mlp: 'MLP (다층 퍼셉트론, GPU)',
  tabm: 'TabM (GPU)',
  tft: 'TFT (Temporal Fusion Transformer, GPU)',
};

export class ExperimentViewController {
  constructor() {
    this.catalog = null;
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
      seed: 'all', // 'all' or 42, 52, 62
      variant: 'bounded', // 'bounded' or 'raw'
      chartMode: 'individual', // 'individual' or 'average'
      filters: {
        facility_id: 'ALL',
        crop_sn: 'ALL',
        sample_num: 'ALL',
        startDate: '',
        endDate: '',
      },
      // Combination filter state
      comboFilter: {
        groupCount: 'ALL', // 'ALL' or 1..7
        includedGroups: new Set(), // Set of 'E1'..'E7'
        searchQuery: '',
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

    // 5. Split buttons (validation / test)
    const splitBtns = document.querySelectorAll('#tab-experiments button[data-split]');
    splitBtns.forEach(btn => {
      btn.addEventListener('click', async () => {
        if (this.state.split === btn.dataset.split) return;
        this.state.split = btn.dataset.split;
        this.updateSplitButtons();
        await this.onSplitChange();
      });
    });

    // 6. Seed select
    const seedSelect = document.getElementById('exp_seed_select');
    if (seedSelect) {
      seedSelect.addEventListener('change', (e) => {
        this.state.seed = e.target.value;
        this.loadPredictionsAndRender();
      });
    }

    // 7. Bounded vs Raw variant toggle
    const variantBtns = document.querySelectorAll('#tab-experiments button[data-variant]');
    variantBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        if (this.state.variant === btn.dataset.variant) return;
        this.state.variant = btn.dataset.variant;
        variantBtns.forEach(b => b.classList.toggle('active', b === btn));
        this.renderComparisonTable();
        this.renderPredictionChart();
      });
    });

    // 8. Refresh experiments button
    const refreshBtn = document.getElementById('exp_refresh_btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<span>⏳</span> 갱신 중...';
        try {
          await refreshExperiments();
          await this.loadInitialData();
          this.showToast('✅ 실험 결과가 최신 상태로 새로고침되었습니다.');
        } catch (err) {
          console.error('[ExperimentView] Refresh failed:', err);
          this.showToast('⚠️ 실험 결과 새로고침에 실패했습니다.');
        } finally {
          refreshBtn.disabled = false;
          refreshBtn.innerHTML = '<span class="refresh-icon">🔄</span> 최신 실험 결과 새로고침';
        }
      });
    }

    // 9. Combination Fast Filters: Group Count pills
    const countPills = document.querySelectorAll('#combo_count_filter_pills .combo-pill');
    countPills.forEach(pill => {
      pill.addEventListener('click', () => {
        countPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        this.state.comboFilter.groupCount = pill.dataset.count;
        this.applyCombinationFilters();
      });
    });

    // 10. Combination Fast Filters: Included Groups pills
    const grpPills = document.querySelectorAll('#combo_included_grp_pills .grp-pill');
    grpPills.forEach(pill => {
      pill.addEventListener('click', () => {
        const grp = pill.dataset.grp;
        if (this.state.comboFilter.includedGroups.has(grp)) {
          this.state.comboFilter.includedGroups.delete(grp);
          pill.classList.remove('active');
        } else {
          this.state.comboFilter.includedGroups.add(grp);
          pill.classList.add('active');
        }
        this.updateResetFiltersButton();
        this.applyCombinationFilters();
      });
    });

    // 11. Reset filters button
    const resetFilterBtn = document.getElementById('btn_reset_combo_filters');
    if (resetFilterBtn) {
      resetFilterBtn.addEventListener('click', () => {
        this.state.comboFilter.groupCount = 'ALL';
        this.state.comboFilter.includedGroups.clear();
        this.state.comboFilter.searchQuery = '';

        // Reset pills UI
        countPills.forEach(p => p.classList.toggle('active', p.dataset.count === 'ALL'));
        grpPills.forEach(p => p.classList.remove('active'));
        const searchInput = document.getElementById('exp_combo_search_input');
        if (searchInput) searchInput.value = '';
        const searchClear = document.getElementById('exp_combo_search_clear');
        if (searchClear) searchClear.style.display = 'none';

        this.updateResetFiltersButton();
        this.applyCombinationFilters();
      });
    }

    // 12. Search input live filtering
    const searchInput = document.getElementById('exp_combo_search_input');
    const searchClear = document.getElementById('exp_combo_search_clear');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.state.comboFilter.searchQuery = e.target.value.trim().toLowerCase();
        if (searchClear) searchClear.style.display = this.state.comboFilter.searchQuery ? 'block' : 'none';
        this.applyCombinationFilters();
      });
    }
    if (searchClear) {
      searchClear.addEventListener('click', () => {
        if (searchInput) searchInput.value = '';
        this.state.comboFilter.searchQuery = '';
        searchClear.style.display = 'none';
        this.applyCombinationFilters();
      });
    }

    // 13. Combination Explorer Modal trigger
    const openExplorerBtn = document.getElementById('exp_open_explorer_btn');
    if (openExplorerBtn) {
      openExplorerBtn.addEventListener('click', () => this.openExplorerModal());
    }

    // 14. Variable Details Modal trigger
    const viewFeaturesBtn = document.getElementById('exp_view_features_btn');
    if (viewFeaturesBtn) {
      viewFeaturesBtn.addEventListener('click', () => this.openVariableModal());
    }

    // 15. Modal close buttons
    const comboModalClose = document.getElementById('combo_modal_close_btn');
    const comboModalConfirm = document.getElementById('combo_modal_confirm_btn');
    if (comboModalClose) comboModalClose.addEventListener('click', () => this.closeExplorerModal());
    if (comboModalConfirm) comboModalConfirm.addEventListener('click', () => this.closeExplorerModal());

    const varModalClose = document.getElementById('modal_close_btn');
    const varModalConfirm = document.getElementById('modal_confirm_btn');
    if (varModalClose) varModalClose.addEventListener('click', () => this.closeVariableModal());
    if (varModalConfirm) varModalConfirm.addEventListener('click', () => this.closeVariableModal());

    // 16. Chart mode toggle (Individual vs Average)
    const modeBtns = document.querySelectorAll('#exp_mode_toggle_group .exp-chart-mode-btn');
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        if (this.state.chartMode === btn.dataset.mode) return;
        this.state.chartMode = btn.dataset.mode;
        modeBtns.forEach(b => b.classList.toggle('active', b === btn));
        this.updateChartFilterStatus();
        this.renderPredictionChart();
      });
    });

    // 17. Chart filter toolbar dropdowns
    const chartFacility = document.getElementById('exp_filter_facility');
    if (chartFacility) {
      chartFacility.addEventListener('change', (e) => {
        this.state.filters.facility_id = e.target.value;
        this.updateChartCropDropdown();
        this.updateChartSampleDropdown();
        this.updateChartFilterStatus();
        this.renderPredictionChart();
      });
    }

    const chartCrop = document.getElementById('exp_filter_crop_sn');
    if (chartCrop) {
      chartCrop.addEventListener('change', (e) => {
        this.state.filters.crop_sn = e.target.value;
        this.updateChartSampleDropdown();
        this.updateChartFilterStatus();
        this.renderPredictionChart();
      });
    }

    const chartSample = document.getElementById('exp_filter_sample_num');
    if (chartSample) {
      chartSample.addEventListener('change', (e) => {
        this.state.filters.sample_num = e.target.value;
        this.updateChartFilterStatus();
        this.renderPredictionChart();
      });
    }

    const chartResetBtn = document.getElementById('exp_filter_reset_btn');
    if (chartResetBtn) {
      chartResetBtn.addEventListener('click', () => {
        this.state.filters.facility_id = 'ALL';
        this.state.filters.crop_sn = 'ALL';
        this.state.filters.sample_num = 'ALL';
        this.state.filters.startDate = '';
        this.state.filters.endDate = '';
        if (chartFacility) chartFacility.value = 'ALL';
        this.updateChartCropDropdown();
        this.updateChartSampleDropdown();
        this.updateChartFilterStatus();
        this.renderPredictionChart();
      });
    }

    // 18. Comparison Table Sorting
    const ths = document.querySelectorAll('#exp_comparison_table th[data-sort]');
    ths.forEach(th => {
      th.addEventListener('click', () => {
        const field = th.dataset.sort;
        if (this.state.sortField === field) {
          this.state.sortAsc = !this.state.sortAsc;
        } else {
          this.state.sortField = field;
          this.state.sortAsc = true;
        }
        this.renderComparisonTable();
      });
    });
  }

  async loadInitialData() {
    try {
      this.catalog = await fetchCatalog();
      this.metadata = await fetchMetadata();

      this.updateTargetDropdown();
      this.populateCombinations();
      await this.loadComparisonData();
      await this.loadPredictionsAndRender();
    } catch (err) {
      console.error('[ExperimentView] Failed to load initial data:', err);
      this.showToast('⚠️ 전수실험 카탈로그 또는 메타데이터를 불러오지 못했습니다.');
    }
  }

  updateTargetDropdown() {
    const targetSelect = document.getElementById('exp_target_select');
    if (!targetSelect) return;
    const targets = CROP_TARGETS[this.state.crop] || [];
    targetSelect.innerHTML = targets.map(t => `<option value="${t.id}">${t.label}</option>`).join('');
    if (!targets.some(t => t.id === this.state.target)) {
      this.state.target = targets[0]?.id || 'tomato_first';
    }
    targetSelect.value = this.state.target;
  }

  populateCombinations() {
    if (!this.catalog || !this.catalog.combinations) return;
    const combos = this.catalog.combinations;

    // Filter combinations
    const filtered = this.getFilteredCombinations(combos);
    const groupSelect = document.getElementById('exp_group_select');
    if (groupSelect) {
      groupSelect.innerHTML = filtered.map(c => {
        const shortNames = c.group_names.map(n => n.replace('기존 ', '')).join(' + ');
        return `<option value="${c.combination_id}">[${c.label}] ${c.group_count}개 군 (${c.feature_count}변수) - ${shortNames}</option>`;
      }).join('');

      // If current selected combo is not in filtered list, pick the first
      if (!filtered.some(c => c.combination_id === this.state.group_id)) {
        this.state.group_id = filtered[0]?.combination_id || combos[0]?.combination_id || '';
      }
      groupSelect.value = this.state.group_id;
    }

    // Update count badge
    const countBadge = document.getElementById('exp_combo_count_badge');
    if (countBadge) {
      countBadge.textContent = `전체 ${combos.length}개 중 ${filtered.length}개 표시`;
    }

    this.updateGroupMetaBar();
  }

  getFilteredCombinations(combos = null) {
    if (!combos) combos = this.catalog?.combinations || [];
    const { groupCount, includedGroups, searchQuery } = this.state.comboFilter;

    return combos.filter(c => {
      // 1. Group Count filter
      if (groupCount !== 'ALL' && c.group_count !== parseInt(groupCount, 10)) {
        return false;
      }
      // 2. Included groups filter (must contain ALL selected groups)
      if (includedGroups.size > 0) {
        for (const grp of includedGroups) {
          if (!c.groups.includes(grp)) return false;
        }
      }
      // 3. Search query
      if (searchQuery) {
        const text = `${c.label} ${c.combination_id} ${c.group_names.join(' ')} ${c.features.map(f => f.feature_name + ' ' + f.desc).join(' ')}`.toLowerCase();
        if (!text.includes(searchQuery)) return false;
      }
      return true;
    });
  }

  applyCombinationFilters() {
    this.populateCombinations();
    this.renderComparisonTable();
  }

  updateResetFiltersButton() {
    const btn = document.getElementById('btn_reset_combo_filters');
    if (!btn) return;
    const hasFilter = this.state.comboFilter.groupCount !== 'ALL' ||
                      this.state.comboFilter.includedGroups.size > 0 ||
                      this.state.comboFilter.searchQuery.length > 0;
    btn.style.display = hasFilter ? 'inline-flex' : 'none';
  }

  updateGroupMetaBar() {
    const combo = this.catalog?.combinations?.find(c => c.combination_id === this.state.group_id);
    if (!combo) return;

    const idEl = document.getElementById('exp_meta_combo_id');
    const featEl = document.getElementById('exp_meta_feature_count');
    const catEl = document.getElementById('exp_meta_categories');
    const winEl = document.getElementById('exp_meta_winner_status');

    if (idEl) idEl.textContent = `${combo.label} (${combo.combination_id})`;
    if (featEl) featEl.textContent = `${combo.feature_count}개 변수`;
    if (catEl) {
      catEl.innerHTML = combo.groups.map(g => {
        const gInfo = this.catalog.variable_groups[g];
        const gName = gInfo ? gInfo.name : g;
        return `<span class="combo-grp-tag"><strong>${g}</strong>: ${gName}</span>`;
      }).join(' ');
    }
    if (winEl) {
      const winner = this.currentComparison?.frozen_winner;
      if (winner === combo.combination_id) {
        winEl.innerHTML = `<span class="badge-winner">🏆 최적 선정 조합</span>`;
      } else {
        winEl.textContent = '-';
      }
    }
  }

  updateSplitButtons() {
    const splitBtns = document.querySelectorAll('#tab-experiments button[data-split]');
    splitBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.split === this.state.split);
    });
  }

  async onTargetOrModelChange() {
    await this.loadComparisonData();
    await this.loadPredictionsAndRender();
  }

  async onGroupChange() {
    this.updateGroupMetaBar();
    this.renderComparisonTable();
    await this.loadPredictionsAndRender();
  }

  async onSplitChange() {
    await this.loadComparisonData();
    await this.loadPredictionsAndRender();
  }

  async loadComparisonData() {
    try {
      this.currentComparison = await fetchComparison(this.state.target, this.state.model, this.state.split);
      this.renderComparisonTable();
      this.updateGroupMetaBar();
    } catch (err) {
      console.error('[ExperimentView] Failed to load comparison:', err);
    }
  }

  renderComparisonTable() {
    const tbody = document.getElementById('exp_comparison_tbody');
    const countBadge = document.getElementById('exp_candidates_count_badge');
    if (!tbody || !this.currentComparison) return;

    const allRows = this.currentComparison.rows || [];
    // Filter rows by combination filter
    const filteredComboIds = new Set(this.getFilteredCombinations().map(c => c.combination_id));
    let rows = allRows.filter(r => filteredComboIds.has(r.group_id));

    if (countBadge) {
      countBadge.textContent = `표시 127개 중 ${rows.length}개`;
    }

    // Sort rows
    const { sortField, sortAsc, variant } = this.state;
    rows.sort((a, b) => {
      let vA, vB;
      if (sortField === 'rank') {
        vA = a.rank ?? 999999;
        vB = b.rank ?? 999999;
      } else if (sortField === 'name') {
        vA = a.readable_name || '';
        vB = b.readable_name || '';
        return sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
      } else if (sortField === 'features') {
        vA = a.feature_count ?? 0;
        vB = b.feature_count ?? 0;
      } else if (sortField === 'seeds') {
        vA = a.completed_seeds ?? 0;
        vB = b.completed_seeds ?? 0;
      } else if (['rmse', 'mae', 'r2', 'ccc'].includes(sortField)) {
        const metricA = a[variant]?.[`${sortField}_mean`];
        const metricB = b[variant]?.[`${sortField}_mean`];
        if (metricA != null && metricB != null) {
          return sortAsc ? metricA - metricB : metricB - metricA;
        }
        if (metricA != null) return -1;
        if (metricB != null) return 1;
        // Both null, sort by group_count asc
        return (a.group_count || 0) - (b.group_count || 0);
      }
      return sortAsc ? vA - vB : vB - vA;
    });

    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="11" class="td-center">필터 조건에 부합하는 조합이 없습니다.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map(r => {
      const isSelected = (r.group_id === this.state.group_id);
      const isWinner = r.is_winner;
      const bMetrics = r[variant] || {};
      const hasResult = (bMetrics.rmse_mean != null);

      const rmseText = hasResult ? `${bMetrics.rmse_mean.toFixed(3)}${bMetrics.rmse_std ? ` ± ${bMetrics.rmse_std.toFixed(3)}` : ''}` : '-';
      const maeText = hasResult ? `${bMetrics.mae_mean.toFixed(3)}${bMetrics.mae_std ? ` ± ${bMetrics.mae_std.toFixed(3)}` : ''}` : '-';
      const r2Text = hasResult && bMetrics.r2_mean != null ? bMetrics.r2_mean.toFixed(3) : '-';
      const cccText = hasResult && bMetrics.ccc_mean != null ? bMetrics.ccc_mean.toFixed(3) : '-';

      const rankBadge = r.rank != null ? `<span class="rank-pill font-bold">#${r.rank}</span>` : `<span class="text-muted">-</span>`;
      const winnerBadge = isWinner ? `<span class="badge-winner">🏆 우승</span>` : `-`;

      const statusTag = hasResult
        ? `<span class="status-tag complete">완료 (${r.completed_seeds}/3)</span>`
        : `<span class="badge-status pending">⏳ 결과 대기 (0/3)</span>`;

      const catBadges = (r.category_ids || []).map(g => `<span class="combo-grp-tag">${g}</span>`).join(' ');

      return `
        <tr class="table-row-selectable ${isSelected ? 'row-active' : ''} ${isWinner ? 'row-winner' : ''}" data-group-id="${r.group_id}">
          <td class="td-center">${rankBadge}</td>
          <td class="td-center">${winnerBadge}</td>
          <td>
            <div class="combo-name-cell">
              <strong>${r.readable_name}</strong>
              <div class="combo-cell-tags">${catBadges} <span class="text-muted text-xs">(${r.group_id})</span></div>
            </div>
          </td>
          <td class="td-center font-bold">${r.feature_count}</td>
          <td class="td-center"><span class="split-pill ${this.state.split}">${this.state.split}</span></td>
          <td class="td-center">${r.completed_seeds}/3</td>
          <td class="td-right font-mono">${rmseText}</td>
          <td class="td-right font-mono">${maeText}</td>
          <td class="td-right font-mono">${r2Text}</td>
          <td class="td-right font-mono">${cccText}</td>
          <td class="td-center">${statusTag}</td>
        </tr>
      `;
    }).join('');

    // Bind row clicks
    tbody.querySelectorAll('tr[data-group-id]').forEach(tr => {
      tr.addEventListener('click', () => {
        const gid = tr.dataset.groupId;
        if (gid && gid !== this.state.group_id) {
          this.state.group_id = gid;
          const groupSelect = document.getElementById('exp_group_select');
          if (groupSelect) groupSelect.value = gid;
          this.onGroupChange();
        }
      });
    });
  }

  async loadPredictionsAndRender() {
    const chartCard = document.getElementById('exp_chart_card');
    const emptyState = document.getElementById('exp_empty_predictions');
    const loadingOverlay = document.getElementById('exp_chart_loading');
    const emptyLabel = document.getElementById('exp_empty_combo_label');

    const currentCombo = this.catalog?.combinations?.find(c => c.combination_id === this.state.group_id);
    const comboLabel = currentCombo ? `${currentCombo.label} (${currentCombo.group_names.join(' + ')})` : this.state.group_id;
    if (emptyLabel) emptyLabel.textContent = comboLabel;

    if (loadingOverlay) loadingOverlay.style.display = 'flex';

    try {
      this.currentPredictions = await fetchPredictions(
        this.state.target,
        this.state.group_id,
        this.state.model,
        this.state.split,
        this.state.seed
      );

      const hasPreds = this.currentPredictions?.predictions && this.currentPredictions.predictions.length > 0;

      if (!hasPreds) {
        if (chartCard) chartCard.style.display = 'none';
        if (emptyState) emptyState.style.display = 'block';
        this.renderEmptyMetrics();
      } else {
        if (chartCard) chartCard.style.display = 'block';
        if (emptyState) emptyState.style.display = 'none';
        this.populateChartFilterDropdowns();
        this.renderPredictionChart();
        this.renderStoredMetrics();
      }
    } catch (err) {
      console.error('[ExperimentView] Failed to load predictions:', err);
      if (chartCard) chartCard.style.display = 'none';
      if (emptyState) emptyState.style.display = 'block';
      this.renderEmptyMetrics();
    } finally {
      if (loadingOverlay) loadingOverlay.style.display = 'none';
    }
  }

  renderEmptyMetrics() {
    const rmseEl = document.getElementById('metric_stored_rmse');
    const maeEl = document.getElementById('metric_stored_mae');
    const r2El = document.getElementById('metric_stored_r2');
    const nEl = document.getElementById('metric_stored_n');
    const paramsEl = document.getElementById('exp_params_json');

    if (rmseEl) rmseEl.textContent = '-';
    if (maeEl) maeEl.textContent = '-';
    if (r2El) r2El.textContent = '-';
    if (nEl) nEl.textContent = '-';
    if (paramsEl) paramsEl.textContent = '// 전수실험 튜닝 결과 대기 중 (실험 미실행)';
  }

  renderStoredMetrics() {
    const meta = this.currentPredictions?.metadata || {};
    const metrics = meta[this.state.variant] || {};

    const rmseEl = document.getElementById('metric_stored_rmse');
    const maeEl = document.getElementById('metric_stored_mae');
    const r2El = document.getElementById('metric_stored_r2');
    const nEl = document.getElementById('metric_stored_n');
    const paramsEl = document.getElementById('exp_params_json');

    if (rmseEl) rmseEl.textContent = metrics.rmse_mean != null ? metrics.rmse_mean.toFixed(3) : '-';
    if (maeEl) maeEl.textContent = metrics.mae_mean != null ? metrics.mae_mean.toFixed(3) : '-';
    if (r2El) r2El.textContent = metrics.r2_mean != null ? metrics.r2_mean.toFixed(3) : '-';
    if (nEl) nEl.textContent = meta.n_eval ? `${meta.n_eval}건` : '-';
    if (paramsEl) {
      paramsEl.textContent = meta.params ? JSON.stringify(meta.params, null, 2) : '// 하이퍼파라미터 정보 없음';
    }
  }

  populateChartFilterDropdowns() {
    const preds = this.currentPredictions?.predictions || [];
    const facilitySelect = document.getElementById('exp_filter_facility');
    if (!facilitySelect) return;

    const facilities = [...new Set(preds.map(p => p.facility_id).filter(Boolean))].sort();
    facilitySelect.innerHTML = '<option value="ALL">전체 시설</option>' +
      facilities.map(f => `<option value="${f}">${f}</option>`).join('');
    facilitySelect.value = this.state.filters.facility_id || 'ALL';

    this.updateChartCropDropdown();
    this.updateChartSampleDropdown();
    this.updateChartFilterStatus();
  }

  updateChartCropDropdown() {
    const preds = this.currentPredictions?.predictions || [];
    const cropSelect = document.getElementById('exp_filter_crop_sn');
    if (!cropSelect) return;

    let sub = preds;
    if (this.state.filters.facility_id !== 'ALL') {
      sub = sub.filter(p => p.facility_id === this.state.filters.facility_id);
    }
    const crops = [...new Set(sub.map(p => String(p.crop_sn)).filter(Boolean))].sort();
    cropSelect.innerHTML = '<option value="ALL">전체 작기 (기간)</option>' +
      crops.map(c => `<option value="${c}">작기 ${c}</option>`).join('');

    if (!crops.includes(this.state.filters.crop_sn)) {
      this.state.filters.crop_sn = 'ALL';
    }
    cropSelect.value = this.state.filters.crop_sn;
  }

  updateChartSampleDropdown() {
    const preds = this.currentPredictions?.predictions || [];
    const sampleSelect = document.getElementById('exp_filter_sample_num');
    if (!sampleSelect) return;

    let sub = preds;
    if (this.state.filters.facility_id !== 'ALL') {
      sub = sub.filter(p => p.facility_id === this.state.filters.facility_id);
    }
    if (this.state.filters.crop_sn !== 'ALL') {
      sub = sub.filter(p => String(p.crop_sn) === this.state.filters.crop_sn);
    }
    const samples = [...new Set(sub.map(p => String(p.sample_num)).filter(Boolean))].sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
    sampleSelect.innerHTML = '<option value="ALL">전체 개체</option>' +
      samples.map(s => `<option value="${s}">개체 ${s}번</option>`).join('');

    if (!samples.includes(this.state.filters.sample_num)) {
      this.state.filters.sample_num = 'ALL';
    }
    sampleSelect.value = this.state.filters.sample_num;
  }

  updateChartFilterStatus() {
    const statusEl = document.getElementById('exp_filter_status');
    if (!statusEl) return;
    const modeText = this.state.chartMode === 'average' ? '일자별 표본 평균 모드' : '개체별 상세 추이 모드';
    const facText = this.state.filters.facility_id === 'ALL' ? '전체 시설' : `시설: ${this.state.filters.facility_id}`;
    const cropText = this.state.filters.crop_sn === 'ALL' ? '전체 작기' : `작기: ${this.state.filters.crop_sn}`;
    const sampleText = this.state.filters.sample_num === 'ALL' ? '전체 개체' : `개체: ${this.state.filters.sample_num}`;

    statusEl.innerHTML = `<span>ℹ️ 현재 조건: <strong>${modeText}</strong> (${facText} / ${cropText} / ${sampleText})</span>`;
  }

  renderPredictionChart() {
    if (!this.chartInstance || !this.currentPredictions?.predictions) return;
    const preds = this.currentPredictions.predictions;
    if (preds.length === 0) return;

    // Apply filtering
    let filtered = preds;
    if (this.state.filters.facility_id !== 'ALL') {
      filtered = filtered.filter(p => p.facility_id === this.state.filters.facility_id);
    }
    if (this.state.filters.crop_sn !== 'ALL') {
      filtered = filtered.filter(p => String(p.crop_sn) === this.state.filters.crop_sn);
    }
    if (this.state.filters.sample_num !== 'ALL') {
      filtered = filtered.filter(p => String(p.sample_num) === this.state.filters.sample_num);
    }

    const valKey = this.state.variant === 'bounded' ? 'prediction_bounded' : 'prediction_raw';

    if (this.state.chartMode === 'average') {
      // Group by target_date and calculate mean
      const dateMap = new Map();
      filtered.forEach(p => {
        const d = p.target_date;
        if (!dateMap.has(d)) dateMap.set(d, { actuals: [], preds: [] });
        if (p.target != null) dateMap.get(d).actuals.push(p.target);
        if (p[valKey] != null) dateMap.get(d).preds.push(p[valKey]);
      });

      const sortedDates = [...dateMap.keys()].sort();
      const actualSeries = sortedDates.map(d => {
        const arr = dateMap.get(d).actuals;
        return arr.length ? +(arr.reduce((a, c) => a + c, 0) / arr.length).toFixed(2) : null;
      });
      const predSeries = sortedDates.map(d => {
        const arr = dateMap.get(d).preds;
        return arr.length ? +(arr.reduce((a, c) => a + c, 0) / arr.length).toFixed(2) : null;
      });

      const option = {
        title: {
          text: `[${this.state.target}] 일자별 실제 관측 평균 vs 모델 예측 평균`,
          subtext: `모델: ${MODEL_LABELS[this.state.model] || this.state.model} | 조합: ${this.state.group_id} | 제약: ${this.state.variant}`,
          left: 10,
          top: 10,
          textStyle: { fontSize: 15, fontWeight: 700, color: '#1e293b' },
          subtextStyle: { fontSize: 12, color: '#64748b' },
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: (params) => {
            const date = params[0]?.axisValueLabel || '';
            let html = `<div style="font-weight:700;margin-bottom:4px;">📅 조사 대상일: ${date}</div>`;
            params.forEach(p => {
              const color = p.color;
              const name = p.seriesName;
              const val = p.value != null ? p.value : '-';
              html += `<div style="display:flex;justify-content:space-between;gap:16px;">
                <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:6px;"></span>${name}</span>
                <span style="font-weight:700;font-family:monospace;">${val}</span>
              </div>`;
            });
            return html;
          }
        },
        legend: {
          data: ['실제 관측 평균', '모델 예측 평균'],
          right: 20,
          top: 15,
        },
        grid: { left: 50, right: 30, top: 75, bottom: 60 },
        xAxis: {
          type: 'category',
          data: sortedDates,
          boundaryGap: false,
          axisLabel: { color: '#64748b', fontSize: 11 },
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#64748b' },
          splitLine: { lineStyle: { color: '#f1f5f9' } },
        },
        dataZoom: [
          { type: 'inside', start: 0, end: 100 },
          { type: 'slider', bottom: 10, height: 20 },
        ],
        series: [
          {
            name: '실제 관측 평균',
            type: 'line',
            data: actualSeries,
            smooth: true,
            symbol: 'circle',
            symbolSize: 6,
            lineStyle: { width: 2.5, color: '#10b981' },
            itemStyle: { color: '#10b981' },
          },
          {
            name: '모델 예측 평균',
            type: 'line',
            data: predSeries,
            smooth: true,
            symbol: 'diamond',
            symbolSize: 6,
            lineStyle: { width: 2.5, color: '#2563eb' },
            itemStyle: { color: '#2563eb' },
          },
        ],
      };
      this.chartInstance.setOption(option, true);
    } else {
      // Individual mode
      const sorted = [...filtered].sort((a, b) => (a.target_date || '').localeCompare(b.target_date || ''));
      const dates = sorted.map(p => p.target_date);
      const actuals = sorted.map(p => p.target);
      const predsData = sorted.map(p => p[valKey]);

      const option = {
        title: {
          text: `[${this.state.target}] 개체별 실제 관측값 vs 모델 예측값 시계열 추이`,
          subtext: `모델: ${MODEL_LABELS[this.state.model] || this.state.model} | 조합: ${this.state.group_id} | 제약: ${this.state.variant}`,
          left: 10,
          top: 10,
          textStyle: { fontSize: 15, fontWeight: 700, color: '#1e293b' },
          subtextStyle: { fontSize: 12, color: '#64748b' },
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: (params) => {
            const idx = params[0]?.dataIndex;
            const p = sorted[idx];
            if (!p) return '';
            let html = `<div style="font-weight:700;margin-bottom:4px;">🏢 ${p.facility_id} | 작기 ${p.crop_sn} | 개체 #${p.sample_num}</div>`;
            html += `<div style="color:#64748b;font-size:11px;margin-bottom:6px;">기준일: ${p.feature_date} → 예측일: ${p.target_date}</div>`;
            params.forEach(param => {
              const val = param.value != null ? param.value : '-';
              html += `<div style="display:flex;justify-content:space-between;gap:16px;">
                <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${param.color};margin-right:6px;"></span>${param.seriesName}</span>
                <span style="font-weight:700;font-family:monospace;">${val}</span>
              </div>`;
            });
            return html;
          }
        },
        legend: {
          data: ['실제 관측값', '모델 예측값'],
          right: 20,
          top: 15,
        },
        grid: { left: 50, right: 30, top: 75, bottom: 60 },
        xAxis: {
          type: 'category',
          data: dates,
          axisLabel: { color: '#64748b', fontSize: 11 },
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#64748b' },
          splitLine: { lineStyle: { color: '#f1f5f9' } },
        },
        dataZoom: [
          { type: 'inside', start: 0, end: 100 },
          { type: 'slider', bottom: 10, height: 20 },
        ],
        series: [
          {
            name: '실제 관측값',
            type: 'line',
            data: actuals,
            symbol: 'circle',
            symbolSize: 6,
            lineStyle: { width: 2, color: '#10b981' },
            itemStyle: { color: '#10b981' },
          },
          {
            name: '모델 예측값',
            type: 'line',
            data: predsData,
            symbol: 'diamond',
            symbolSize: 6,
            lineStyle: { width: 2, color: '#2563eb' },
            itemStyle: { color: '#2563eb' },
          },
        ],
      };
      this.chartInstance.setOption(option, true);
    }
  }

  openExplorerModal() {
    const modal = document.getElementById('exp_combo_explorer_modal');
    if (!modal) return;
    modal.style.display = 'flex';
    this.renderModalCards();
    this.bindModalEvents();
  }

  closeExplorerModal() {
    const modal = document.getElementById('exp_combo_explorer_modal');
    if (modal) modal.style.display = 'none';
  }

  renderModalCards() {
    const container = document.getElementById('modal_combo_cards_grid');
    const countBadge = document.getElementById('modal_combo_count_badge');
    if (!container || !this.catalog) return;

    const combos = this.catalog.combinations || [];
    const filtered = this.getFilteredCombinations(combos);

    if (countBadge) {
      countBadge.textContent = `${combos.length}개 중 ${filtered.length}개 표시`;
    }

    if (filtered.length === 0) {
      container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 40px; color:#64748b;">조건에 맞는 변수군 조합이 없습니다.</div>`;
      return;
    }

    container.innerHTML = filtered.map(c => {
      const isSelected = (c.combination_id === this.state.group_id);
      const grpTags = c.groups.map(g => {
        const gName = this.catalog.variable_groups[g]?.name || g;
        return `<span class="combo-grp-tag" title="${gName}"><strong>${g}</strong></span>`;
      }).join(' ');

      return `
        <div class="combo-card ${isSelected ? 'selected' : ''}" data-combo-id="${c.combination_id}">
          <div class="combo-card-header">
            <span class="combo-card-label"><span>🧩</span> ${c.label}</span>
            <span class="combo-card-badge">${c.group_count}개 군</span>
          </div>
          <div class="combo-card-groups">${grpTags}</div>
          <div style="font-size: 12px; color: #475569; line-height: 1.4;">${c.group_names.join(' + ')}</div>
          <div class="combo-card-footer">
            <span class="feat-count">총 ${c.feature_count}개 변수</span>
            <button type="button" class="btn-select-combo">${isSelected ? '선택됨 ✓' : '선택'}</button>
          </div>
        </div>
      `;
    }).join('');

    // Bind card clicks
    container.querySelectorAll('.combo-card').forEach(card => {
      card.addEventListener('click', () => {
        const cid = card.dataset.comboId;
        if (cid) {
          this.state.group_id = cid;
          const groupSelect = document.getElementById('exp_group_select');
          if (groupSelect) groupSelect.value = cid;
          this.onGroupChange();
          this.closeExplorerModal();
        }
      });
    });
  }

  bindModalEvents() {
    // Modal group count pills
    const pills = document.querySelectorAll('#modal_combo_count_pills .combo-pill');
    pills.forEach(pill => {
      pill.onclick = () => {
        pills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        this.state.comboFilter.groupCount = pill.dataset.count;
        this.renderModalCards();
      };
    });

    // Modal included group pills
    const grpPills = document.querySelectorAll('#modal_combo_grp_pills .grp-pill');
    const resetBtn = document.getElementById('modal_btn_reset_filters');
    grpPills.forEach(pill => {
      pill.onclick = () => {
        const grp = pill.dataset.grp;
        if (this.state.comboFilter.includedGroups.has(grp)) {
          this.state.comboFilter.includedGroups.delete(grp);
          pill.classList.remove('active');
        } else {
          this.state.comboFilter.includedGroups.add(grp);
          pill.classList.add('active');
        }
        if (resetBtn) resetBtn.style.display = this.state.comboFilter.includedGroups.size > 0 ? 'inline-flex' : 'none';
        this.renderModalCards();
      };
    });

    if (resetBtn) {
      resetBtn.onclick = () => {
        this.state.comboFilter.includedGroups.clear();
        grpPills.forEach(p => p.classList.remove('active'));
        resetBtn.style.display = 'none';
        this.renderModalCards();
      };
    }

    // Modal search
    const searchInput = document.getElementById('modal_combo_search_input');
    const searchClear = document.getElementById('modal_combo_search_clear');
    if (searchInput) {
      searchInput.oninput = (e) => {
        this.state.comboFilter.searchQuery = e.target.value.trim().toLowerCase();
        if (searchClear) searchClear.style.display = this.state.comboFilter.searchQuery ? 'block' : 'none';
        this.renderModalCards();
      };
    }
    if (searchClear) {
      searchClear.onclick = () => {
        if (searchInput) searchInput.value = '';
        this.state.comboFilter.searchQuery = '';
        searchClear.style.display = 'none';
        this.renderModalCards();
      };
    }
  }

  openVariableModal() {
    const modal = document.getElementById('exp_variable_modal');
    if (!modal || !this.catalog) return;

    const combo = this.catalog.combinations?.find(c => c.combination_id === this.state.group_id);
    if (!combo) return;

    const nameEl = document.getElementById('modal_group_name');
    const readableEl = document.getElementById('modal_group_readable');
    const idEl = document.getElementById('modal_group_id');
    const countEl = document.getElementById('modal_group_count');
    const catListEl = document.getElementById('modal_categories_list');
    const featuresContainer = document.getElementById('modal_features_container');

    if (nameEl) nameEl.textContent = `조합 [${combo.label}] 구성 변수 상세`;
    if (readableEl) readableEl.textContent = combo.display_name;
    if (idEl) idEl.textContent = combo.combination_id;
    if (countEl) countEl.textContent = `${combo.feature_count}개 변수 (${combo.group_count}개 변수군)`;
    if (catListEl) {
      catListEl.innerHTML = combo.groups.map(g => {
        const gName = this.catalog.variable_groups[g]?.name || g;
        return `<span class="combo-grp-tag"><strong>${g}</strong>: ${gName}</span>`;
      }).join(' ');
    }

    if (featuresContainer) {
      let html = '';

      // 1. Fixed Context C0 Section
      const c0 = this.catalog.variable_groups['C0'];
      if (c0) {
        html += `
          <div class="modal-feature-group" style="margin-bottom: 20px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
              <h5 style="font-size:14px;font-weight:700;color:#0f172a;">🔒 C0: ${c0.name} (고정 공통 메타)</h5>
              <span class="badge-vrole">고정 분할/정렬 키</span>
            </div>
            <p style="font-size:12px;color:#64748b;margin-bottom:10px;">${c0.description}</p>
            <table class="variable-table">
              <thead>
                <tr>
                  <th>변수 식별명</th>
                  <th>컬럼명</th>
                  <th>설명</th>
                  <th>데이터 타입</th>
                </tr>
              </thead>
              <tbody>
                ${c0.features.map(f => `
                  <tr>
                    <td class="font-mono font-bold">${f.feature_id}</td>
                    <td class="font-mono">${f.feature_name}</td>
                    <td>${f.desc}</td>
                    <td><span class="badge-vtype">${f.type}</span></td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
      }

      // 2. Included Variable Groups
      combo.groups.forEach(gCode => {
        const grp = this.catalog.variable_groups[gCode];
        if (!grp) return;

        html += `
          <div class="modal-feature-group" style="margin-bottom: 20px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
              <h5 style="font-size:14px;font-weight:700;color:#0f172a;">🧩 ${gCode}: ${grp.name} (${grp.feature_count}개 변수)</h5>
              <span class="combo-card-badge">${gCode}</span>
            </div>
            <table class="variable-table">
              <thead>
                <tr>
                  <th>변수 식별명</th>
                  <th>컬럼명</th>
                  <th>설명</th>
                  <th>데이터 타입</th>
                  <th>소속 테이블</th>
                  <th>실험 역할</th>
                  <th>비고</th>
                </tr>
              </thead>
              <tbody>
                ${grp.features.map(f => `
                  <tr>
                    <td class="font-mono font-bold">${f.feature_id}</td>
                    <td class="font-mono">${f.feature_name}</td>
                    <td>${f.desc}</td>
                    <td><span class="badge-vtype">${f.type}</span></td>
                    <td>${f.source_table || '-'}</td>
                    <td><span class="badge-vrole">${f.role || '후보입력'}</span></td>
                    <td class="text-xs text-muted">${f.note || '-'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
      });

      featuresContainer.innerHTML = html;
    }

    modal.style.display = 'flex';
  }

  closeVariableModal() {
    const modal = document.getElementById('exp_variable_modal');
    if (modal) modal.style.display = 'none';
  }

  showToast(msg) {
    const toast = document.getElementById('alert_toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = 'flex';
    setTimeout(() => { toast.style.display = 'none'; }, 3500);
  }
}
