/**
 * analysis_view.js
 * Controller for Tab 3: 생육 예측 분석 (Growth Prediction & AX Direction Analysis)
 *
 * Implements:
 * 1. Three internal sub-tabs:
 *    - anl-subtab-perf: 성능·오차 분석 (Observation-style Toolbar, 4 KPI cards, ECharts visual models, Evidence accordion)
 *    - anl-subtab-ax: AX 데이터 구축 방향 (H01~H07 Master-Detail, 88 fields Dynamic Schema explorer, 5 Principles)
 *    - anl-subtab-kpi: 검증 계획·KPI (B0~ABL Step Pipeline, Target KPI planning card, Full 8-target reference table)
 * 2. ECharts chart instances with clean lifecycle & resize handlers
 * 3. Dynamic schema table counting and Korean display names (preserving true DB keys)
 * 4. Strict numerical integrity (no fabricated numbers, explicit unexperimented / planning proposal badges)
 */

import { fetchAnalysisBundle } from './analysis_api.js';

export const TARGET_META = {
  tomato_first: { crop: 'tomato', name: '토마토 1화방 꽃수', icon: '🍅', truss: '1화방' },
  tomato_second: { crop: 'tomato', name: '토마토 2화방 꽃수', icon: '🍅', truss: '2화방' },
  tomato_third: { crop: 'tomato', name: '토마토 3화방 꽃수', icon: '🍅', truss: '3화방' },
  tomato_sum123: { crop: 'tomato', name: '토마토 1~3화방 합계 꽃수', icon: '🍅', truss: '합계' },
  strawberry_first: { crop: 'strawberry', name: '딸기 1화방 착과수', icon: '🍓', truss: '1화방' },
  strawberry_second: { crop: 'strawberry', name: '딸기 2화방 착과수', icon: '🍓', truss: '2화방' },
  strawberry_third: { crop: 'strawberry', name: '딸기 3화방 착과수', icon: '🍓', truss: '3화방' },
  strawberry_sum123: { crop: 'strawberry', name: '딸기 1~3화방 합계 착과수', icon: '🍓', truss: '합계' },
};

export const DIMENSION_META = {
  all: { label: '전체 (All)', posthoc: false },
  facility: { label: '🏢 시설별 (Facility)', posthoc: false },
  crop_cycle: { label: '🌱 작기별 (Crop Cycle)', posthoc: false },
  crop_age_days: { label: '📅 작기 경과일 (Crop Age)', posthoc: false },
  previous_gap_days: { label: '⏱️ 직전 조사 간격 (Prev Gap)', posthoc: false },
  horizon_days: { label: '⌛ 다음 실제 간격 [사후 진단]', posthoc: true },
  change: { label: '📈 수량 변화 구간 [사후 진단]', posthoc: true },
  label_positive: { label: '➕ 정답 양수 여부 [사후 진단]', posthoc: true },
  current_positive: { label: '🔍 현재값 양수 여부', posthoc: false },
  selected_input_missing: { label: '⚠️ 입력 결측 여부 (Missing)', posthoc: false },
};

export const SCHEMA_TABLE_META = {
  all: { label: '전체', desc: '모든 AX 제안 필드' },
  ax_event: { label: '작업 내역 (ax_event)', desc: '적화, 적과, 수확, 방제 등 인위적 작업' },
  ax_observation: { label: '추가 관측 (ax_observation)', desc: '화방별 개화·착과 상태 및 생체 지표' },
  ax_context: { label: '온실 맥락 (ax_context)', desc: '시설 구조, 센서 스펙, 환경 이상' },
  ax_constraint: { label: '작업 제약 (ax_constraint)', desc: '작업 지연 사유 및 설비 제약' },
  survey_schedule: { label: '조사 일정 (survey_schedule)', desc: '실제 조사 주기 및 간격 일정' },
  existing_growth: { label: '기존 생육 (existing_growth)', desc: '초장, 엽수 등 기존 원장 지표' },
};

export class AnalysisViewController {
  constructor() {
    this.bundle = null;
    this.state = {
      crop: 'tomato',
      target: 'tomato_first',
      split: 'test',
      dimension: 'facility',
      activeSubTab: 'anl-subtab-perf',
      selectedHypId: 'H01',
      selectedArmId: 'B0',
      schemaTableFilter: 'all',
      schemaSearchQuery: '',
    };

    // ECharts instances
    this.chartModelComp = null;
    this.chartErrorStrata = null;
    this.chartSseShare = null;
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;
    try {
      this.bundle = await fetchAnalysisBundle();
      this.bindEvents();
      this.initDimensionDropdown();
      this.initSchemaTabs();
      this.render();
      this.initCharts();
      this.initialized = true;

      // Global resize listener
      window.addEventListener('resize', () => this.resizeCharts());
    } catch (err) {
      console.error('Failed to initialize Analysis View:', err);
      const container = document.getElementById('tab-analysis');
      if (container) {
        container.innerHTML = `<div class="guide-banner" style="border-left-color: #ef4444;"><div class="guide-icon">⚠️</div><div class="guide-text"><h4>분석 데이터를 불러오지 못했습니다</h4><p>${err.message}</p></div></div>`;
      }
    }
  }

  // =========================================================================
  // EVENT BINDINGS
  // =========================================================================
  bindEvents() {
    // 1. Sub-tab Navigation
    const subnavBtns = document.querySelectorAll('.anl-subnav-btn');
    subnavBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetSubtab = btn.dataset.subtab;
        if (targetSubtab === this.state.activeSubTab) return;

        subnavBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.anl-subtab-pane').forEach(pane => {
          pane.classList.remove('active');
        });
        const activePane = document.getElementById(targetSubtab);
        if (activePane) activePane.classList.add('active');

        this.state.activeSubTab = targetSubtab;

        // Resize charts if entering performance subtab
        if (targetSubtab === 'anl-subtab-perf') {
          setTimeout(() => this.resizeCharts(), 50);
        }
      });
    });

    // 2. Crop Selector Radio
    const cropRadios = document.querySelectorAll('input[name="anl_crop_radio"]');
    cropRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.state.crop = e.target.value;
        // Auto-select first target of selected crop
        const targets = Object.keys(TARGET_META).filter(k => TARGET_META[k].crop === this.state.crop);
        if (targets.length > 0) this.state.target = targets[0];
        this.updateTargetDropdownOptions();
        this.renderTargetDependentViews();
      });
    });

    // 3. Target Dropdown
    const targetSelect = document.getElementById('anl_target_select');
    if (targetSelect) {
      targetSelect.addEventListener('change', (e) => {
        this.state.target = e.target.value;
        const cropInfo = TARGET_META[this.state.target];
        if (cropInfo && cropInfo.crop !== this.state.crop) {
          this.state.crop = cropInfo.crop;
          const radio = document.querySelector(`input[name="anl_crop_radio"][value="${cropInfo.crop}"]`);
          if (radio) radio.checked = true;
        }
        this.renderTargetDependentViews();
      });
    }

    // 4. Split Toggle Buttons
    const splitBtns = document.querySelectorAll('.anl-split-btn');
    splitBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        splitBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.state.split = btn.dataset.split;
        this.renderTargetDependentViews();
      });
    });

    // 5. Dimension Dropdown
    const dimSelect = document.getElementById('anl_dimension_select');
    if (dimSelect) {
      dimSelect.addEventListener('change', (e) => {
        this.state.dimension = e.target.value;
        this.renderStrataVisuals();
      });
    }

    // 6. Schema Search Input
    const schemaSearch = document.getElementById('anl_schema_search');
    if (schemaSearch) {
      schemaSearch.addEventListener('input', (e) => {
        this.state.schemaSearchQuery = e.target.value.trim().toLowerCase();
        this.renderSchemaTable();
      });
    }
  }

  // =========================================================================
  // DROPDOWN & TABS INITIALIZERS
  // =========================================================================
  updateTargetDropdownOptions() {
    const targetSelect = document.getElementById('anl_target_select');
    if (!targetSelect) return;
    targetSelect.innerHTML = '';

    const targets = Object.keys(TARGET_META).filter(k => TARGET_META[k].crop === this.state.crop);
    targets.forEach(tId => {
      const opt = document.createElement('option');
      opt.value = tId;
      opt.textContent = `${TARGET_META[tId].icon} ${TARGET_META[tId].name}`;
      if (tId === this.state.target) opt.selected = true;
      targetSelect.appendChild(opt);
    });
  }

  initDimensionDropdown() {
    const dimSelect = document.getElementById('anl_dimension_select');
    if (!dimSelect) return;
    dimSelect.innerHTML = '';

    Object.keys(DIMENSION_META).forEach(k => {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = DIMENSION_META[k].label;
      if (k === this.state.dimension) opt.selected = true;
      dimSelect.appendChild(opt);
    });
  }

  initSchemaTabs() {
    const tabsContainer = document.getElementById('anl_schema_tabs');
    if (!tabsContainer || !this.bundle) return;
    const schema = this.bundle.tables.ax_field_schema || [];

    // Calculate true dynamic counts per table
    const counts = {};
    schema.forEach(row => {
      counts[row.table] = (counts[row.table] || 0) + 1;
    });

    const categories = ['all', 'ax_event', 'ax_observation', 'ax_context', 'ax_constraint', 'survey_schedule', 'existing_growth'];

    tabsContainer.innerHTML = categories.map(cat => {
      const cnt = (cat === 'all') ? schema.length : (counts[cat] || 0);
      const meta = SCHEMA_TABLE_META[cat] || { label: cat };
      const activeClass = (cat === this.state.schemaTableFilter) ? 'active' : '';
      const displayLabel = cat === 'all' ? `전체 (${cnt})` : `${meta.label.split(' ')[0]} (${cnt})`;
      return `
        <button type="button" class="anl-schema-tab-btn ${activeClass}" data-table="${cat}">
          ${displayLabel}
        </button>
      `;
    }).join('');

    // Tab click binding
    tabsContainer.querySelectorAll('.anl-schema-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        tabsContainer.querySelectorAll('.anl-schema-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.state.schemaTableFilter = btn.dataset.table;
        this.renderSchemaTable();
      });
    });
  }

  // =========================================================================
  // MAIN RENDER DISPATCHER
  // =========================================================================
  render() {
    this.updateTargetDropdownOptions();
    this.renderTargetDependentViews();

    // Subtab 2: AX Roadmaps
    this.renderHypothesesMasterDetail();
    this.renderSchemaTable();

    // Subtab 3: Validation Protocol & KPI
    this.renderComparisonArmsPipeline();
    this.renderTargetKpiCard();
    this.renderFullKpiTable();
  }

  renderTargetDependentViews() {
    this.renderKpiSummaryCards();
    this.renderDiagnosisCard();
    this.renderBaselineComparisonTable();
    this.renderCrossedStrataTable();
    this.renderStrataVisuals();
    this.renderTargetKpiCard();
  }

  // =========================================================================
  // SUBTAB 1: PERFORMANCE & ERROR ANALYSIS
  // =========================================================================

  // 1-1. 4 KPI Summary Cards
  renderKpiSummaryCards() {
    if (!this.bundle) return;
    const { baseline_performance, selected_models, cohort_profile } = this.bundle.tables;

    const baseRow = (baseline_performance || []).find(
      r => r.target === this.state.target && r.split === this.state.split
    );
    const selModel = (selected_models || []).find(r => r.target === this.state.target);
    const cohort = (cohort_profile || []).find(
      r => r.target === this.state.target && r.split === this.state.split
    );

    if (!baseRow) return;

    // Filter status bar
    const modelEl = document.getElementById('anl_summary_model');
    const groupEl = document.getElementById('anl_summary_group');
    const nEl = document.getElementById('anl_summary_n');
    const facEl = document.getElementById('anl_summary_facilities');

    if (modelEl) modelEl.textContent = (baseRow.model || '').toUpperCase();
    if (groupEl) groupEl.textContent = `${TARGET_META[this.state.target].name} [${baseRow.group_id}]`;
    if (nEl) nEl.textContent = `N=${baseRow.n.toLocaleString()}건`;
    if (facEl) facEl.textContent = `시설 ${cohort ? cohort.facilities : baseRow.facilities}개소`;

    // 4 Cards
    const rmseEl = document.getElementById('anl_summary_rmse');
    const rmseSub = document.getElementById('anl_summary_rmse_sub');
    const maeEl = document.getElementById('anl_summary_mae');
    const maeSub = document.getElementById('anl_summary_mae_sub');
    const r2El = document.getElementById('anl_summary_r2');
    const r2Sub = document.getElementById('anl_summary_r2_sub');
    const cccEl = document.getElementById('anl_summary_ccc');
    const cccSub = document.getElementById('anl_summary_ccc_sub');

    if (rmseEl) rmseEl.textContent = baseRow.rmse_mean.toFixed(4);
    if (rmseSub) {
      const sd = baseRow.rmse_seed_std !== null && baseRow.rmse_seed_std !== undefined ? `seed 변동 ± ${baseRow.rmse_seed_std.toFixed(4)}` : '단일 seed';
      rmseSub.textContent = sd;
    }

    if (maeEl) maeEl.textContent = baseRow.mae_mean.toFixed(4);
    if (maeSub) {
      const sd = baseRow.mae_seed_std !== null && baseRow.mae_seed_std !== undefined ? `seed 변동 ± ${baseRow.mae_seed_std.toFixed(4)}` : '단일 seed';
      maeSub.textContent = sd;
    }

    if (r2El) {
      if (baseRow.r2_mean !== null && !isNaN(baseRow.r2_mean)) {
        r2El.textContent = baseRow.r2_mean.toFixed(4);
        if (r2Sub) r2Sub.textContent = baseRow.r2_mean < 0 ? '음수 R²: 분산 대비 오차 반영' : '설명력 지표';
      } else {
        r2El.textContent = '미정의';
        if (r2Sub) r2Sub.textContent = '분산 0 또는 계산 불가';
      }
    }

    if (cccEl) {
      if (baseRow.ccc_mean !== null && !isNaN(baseRow.ccc_mean)) {
        cccEl.textContent = baseRow.ccc_mean.toFixed(4);
        if (cccSub) cccSub.textContent = '실제-예측 일치상관계수';
      } else {
        cccEl.textContent = '-';
        if (cccSub) cccSub.textContent = '산출 불가';
      }
    }
  }

  // 1-2. Charts Rendering
  initCharts() {
    const domModel = document.getElementById('anl_chart_model_comp');
    const domStrata = document.getElementById('anl_chart_strata');
    const domSse = document.getElementById('anl_chart_sse');

    if (domModel && window.echarts) {
      this.chartModelComp = echarts.init(domModel);
    }
    if (domStrata && window.echarts) {
      this.chartErrorStrata = echarts.init(domStrata);
    }
    if (domSse && window.echarts) {
      this.chartSseShare = echarts.init(domSse);
    }

    this.renderCharts();
  }

  resizeCharts() {
    if (this.chartModelComp) this.chartModelComp.resize();
    if (this.chartErrorStrata) this.chartErrorStrata.resize();
    if (this.chartSseShare) this.chartSseShare.resize();
  }

  renderCharts() {
    this.renderModelComparisonChart();
    this.renderStrataVisuals();
  }

  // Chart 1: Model Comparison Bar Chart
  renderModelComparisonChart() {
    if (!this.chartModelComp || !this.bundle) return;
    const baseRow = (this.bundle.tables.baseline_performance || []).find(
      r => r.target === this.state.target && r.split === this.state.split
    );
    if (!baseRow) return;

    const mlRmse = baseRow.rmse_mean;
    const persRmse = baseRow.persistence_rmse || 0;
    const zeroRmse = baseRow.zero_rmse || 0;

    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          const p = params[0];
          let extra = '';
          if (p.name.includes('선정 모델')) {
            extra = '<br><span style="color:#2563eb;font-size:11px;">※ 검증 동결 1순위 최적 모델</span>';
          } else if (p.name.includes('유지')) {
            const diff = ((mlRmse - persRmse) / persRmse * 100).toFixed(1);
            extra = `<br><span style="color:#64748b;font-size:11px;">현재 수량 유지 대비 ${diff > 0 ? '+' + diff + '% 악화' : diff + '% 개선'}</span>`;
          } else if (p.name.includes('0 예측')) {
            extra = '<br><span style="color:#f59e0b;font-size:11px;">※ 불균형 정답(96%가 0)으로 인한 착시, 실제 대안 불가</span>';
          }
          return `<strong>${p.name}</strong><br>RMSE: <strong>${p.value.toFixed(4)}</strong> (개수)${extra}`;
        }
      },
      grid: {
        top: 25,
        bottom: 25,
        left: 140,
        right: 40,
      },
      xAxis: {
        type: 'value',
        name: 'RMSE (개수)',
        nameTextStyle: { fontSize: 11, color: '#64748b' },
        axisLabel: { color: '#64748b', fontSize: 11 },
        splitLine: { lineStyle: { color: '#f1f5f9' } }
      },
      yAxis: {
        type: 'category',
        inverse: true,
        data: [
          `선정 모델 (${(baseRow.model || '').toUpperCase()})`,
          '현재값 유지 (Persistence)',
          '항상 0 예측 (Zero Predictor)'
        ],
        axisLabel: { color: '#1e293b', fontSize: 12, fontWeight: 600 },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#cbd5e1' } }
      },
      series: [{
        name: 'RMSE',
        type: 'bar',
        barWidth: 26,
        data: [
          { value: mlRmse, itemStyle: { color: '#2563eb', borderRadius: [0, 4, 4, 0] } },
          { value: persRmse, itemStyle: { color: '#64748b', borderRadius: [0, 4, 4, 0] } },
          { value: zeroRmse, itemStyle: { color: '#f59e0b', borderRadius: [0, 4, 4, 0] } }
        ],
        label: {
          show: true,
          position: 'right',
          formatter: (p) => p.value.toFixed(4),
          color: '#0f172a',
          fontWeight: 700,
          fontSize: 12
        }
      }]
    };

    this.chartModelComp.setOption(option, true);
  }

  // Chart 2 & 3: Strata RMSE & SSE Share Charts + Strata Table
  renderStrataVisuals() {
    if (!this.bundle) return;
    const { error_strata } = this.bundle.tables;

    // Filter rows for current target, split, and dimension
    const rows = (error_strata || []).filter(
      r => r.target === this.state.target && r.split === this.state.split && r.dimension === this.state.dimension
    );

    // Update subtitle
    const subTitle = document.getElementById('anl_chart_strata_subtitle');
    const dimMeta = DIMENSION_META[this.state.dimension] || { label: this.state.dimension };
    if (subTitle) subTitle.textContent = `${dimMeta.label} 구간별 RMSE 분포 (${rows.length}개 구간)`;

    // Render Chart 2 (Strata RMSE Bar)
    if (this.chartErrorStrata) {
      if (rows.length === 0) {
        this.chartErrorStrata.clear();
      } else {
        const categories = rows.map(r => r.level);
        const rmseData = rows.map(r => r.rmse_mean);

        const option = {
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: (params) => {
              const idx = params[0].dataIndex;
              const r = rows[idx];
              const smallBadge = r.small_n ? '<br><span style="color:#ef4444;font-size:11px;">⚠️ 소표본 (N<30)</span>' : '';
              return `<strong>${r.level}</strong><br>표본 N: ${r.n}건 (시설 ${r.facilities})<br>RMSE: <strong>${r.rmse_mean.toFixed(4)}</strong><br>MAE: ${r.mae_mean.toFixed(4)}${smallBadge}`;
            }
          },
          grid: { top: 20, bottom: 40, left: 50, right: 20 },
          xAxis: {
            type: 'category',
            data: categories,
            axisLabel: {
              interval: 0,
              rotate: categories.length > 5 ? 25 : 0,
              fontSize: 11,
              color: '#475569'
            },
            axisTick: { alignWithLabel: true }
          },
          yAxis: {
            type: 'value',
            name: 'RMSE',
            nameTextStyle: { fontSize: 11, color: '#64748b' },
            axisLabel: { color: '#64748b', fontSize: 11 },
            splitLine: { lineStyle: { color: '#f1f5f9' } }
          },
          series: [{
            name: 'RMSE',
            type: 'bar',
            barMaxWidth: 38,
            data: rmseData.map((val, i) => {
              const r = rows[i];
              // Highlight high error concentration or small N
              const color = r.small_n ? '#f87171' : (r.squared_error_share > r.row_share * 1.5 ? '#e11d48' : '#3b82f6');
              return { value: val, itemStyle: { color, borderRadius: [4, 4, 0, 0] } };
            }),
            label: {
              show: true,
              position: 'top',
              formatter: (p) => p.value.toFixed(2),
              fontSize: 10.5,
              color: '#475569'
            }
          }]
        };
        this.chartErrorStrata.setOption(option, true);
      }
    }

    // Render Chart 3 (Sample Share % vs SSE Share %)
    if (this.chartSseShare) {
      if (rows.length === 0) {
        this.chartSseShare.clear();
      } else {
        const categories = rows.map(r => r.level);
        const rowShareData = rows.map(r => +(r.row_share * 100).toFixed(1));
        const sseShareData = rows.map(r => +(r.squared_error_share * 100).toFixed(1));

        const option = {
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: (params) => {
              const idx = params[0].dataIndex;
              const r = rows[idx];
              const isHigh = r.squared_error_share > r.row_share * 1.5;
              const highWarn = isHigh ? '<br><span style="color:#ef4444;font-weight:700;">⚠️ 오차 집중 구간 (SSE % > 표본 % * 1.5)</span>' : '';
              return `<strong>${r.level}</strong><br>표본 N: ${r.n}건<br>표본 비중: <strong>${(r.row_share * 100).toFixed(1)}%</strong><br>제곱오차 비중 (SSE): <strong>${(r.squared_error_share * 100).toFixed(1)}%</strong>${highWarn}`;
            }
          },
          legend: {
            top: 0,
            right: 10,
            itemWidth: 12,
            itemHeight: 12,
            textStyle: { fontSize: 11.5, color: '#475569' }
          },
          grid: { top: 35, bottom: 40, left: 50, right: 20 },
          xAxis: {
            type: 'category',
            data: categories,
            axisLabel: {
              interval: 0,
              rotate: categories.length > 5 ? 25 : 0,
              fontSize: 11,
              color: '#475569'
            }
          },
          yAxis: {
            type: 'value',
            name: '비중 (%)',
            nameTextStyle: { fontSize: 11, color: '#64748b' },
            axisLabel: { formatter: '{value}%', color: '#64748b', fontSize: 11 },
            max: (val) => Math.min(100, Math.ceil(val.max / 10) * 10 + 10),
            splitLine: { lineStyle: { color: '#f1f5f9' } }
          },
          series: [
            {
              name: '표본 비중 (%)',
              type: 'bar',
              barMaxWidth: 24,
              data: rowShareData,
              itemStyle: { color: '#93c5fd', borderRadius: [3, 3, 0, 0] }
            },
            {
              name: '제곱오차 비중 (SSE %)',
              type: 'bar',
              barMaxWidth: 24,
              data: sseShareData,
              itemStyle: { color: '#ef4444', borderRadius: [3, 3, 0, 0] }
            }
          ]
        };
        this.chartSseShare.setOption(option, true);
      }
    }

    // Render Strata Table in Accordion
    const tbody = document.getElementById('anl_strata_tbody');
    if (tbody) {
      if (rows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="td-center text-muted">선택한 조건의 구간별 오차 데이터가 없습니다.</td></tr>`;
      } else {
        tbody.innerHTML = rows.map(r => {
          const smallNBadge = r.small_n ? `<span class="anl-badge-small-n" title="표본 30건 미만 소표본">소표본</span>` : '';
          const posthocBadge = r.posthoc_label_based ? `<span class="anl-badge-posthoc" title="미래 정답 기반 사후 진단">사후</span>` : '';
          const isConcentrated = (r.squared_error_share > r.row_share * 1.5 && r.squared_error_share > 0.3);

          const r2Text = (r.r2_mean !== null && !isNaN(r.r2_mean)) ? r.r2_mean.toFixed(4) : '<span class="text-muted">미정의</span>';
          const cccText = (r.ccc_mean !== null && !isNaN(r.ccc_mean)) ? r.ccc_mean.toFixed(4) : '-';
          const biasText = (r.bias_mean !== null && !isNaN(r.bias_mean)) ? (r.bias_mean > 0 ? `+${r.bias_mean.toFixed(4)}` : r.bias_mean.toFixed(4)) : '-';

          return `
            <tr class="${isConcentrated ? 'row-alert' : ''}">
              <td><strong>${r.level}</strong> ${smallNBadge} ${posthocBadge}</td>
              <td class="td-right">${r.n}</td>
              <td class="td-right">${r.facilities}</td>
              <td class="td-right">${(r.row_share * 100).toFixed(1)}%</td>
              <td class="td-right ${isConcentrated ? 'text-danger font-bold' : ''}">${(r.squared_error_share * 100).toFixed(1)}%</td>
              <td class="td-right font-bold">${r.rmse_mean.toFixed(4)}</td>
              <td class="td-right">${r.mae_mean.toFixed(4)}</td>
              <td class="td-right">${r2Text}</td>
              <td class="td-right">${cccText}</td>
              <td class="td-right">${biasText}</td>
            </tr>
          `;
        }).join('');
      }
    }
  }

  // 1-3. Target Diagnosis Box
  renderDiagnosisCard() {
    if (!this.bundle) return;
    const { target_diagnosis } = this.bundle.tables;
    const item = (target_diagnosis || []).find(r => r.target === this.state.target);
    const container = document.getElementById('anl_diagnosis_content');
    if (!container) return;

    if (!item) {
      container.innerHTML = `<p class="text-muted">해당 타깃의 진단 리포트가 없습니다.</p>`;
      return;
    }

    container.innerHTML = `
      <div class="anl-diag-box">
        <div class="anl-diag-section">
          <div class="anl-diag-lbl">🔍 핵심 관측 사실 (Finding)</div>
          <p class="anl-diag-val">${item.finding}</p>
        </div>
        <div class="anl-diag-section" style="margin-top: 10px;">
          <div class="anl-diag-lbl">🎯 후속 검증 방향 (Next Step)</div>
          <p class="anl-diag-val highlight">${item.next_validation}</p>
        </div>
        <div class="anl-diag-footer">
          <span>📑 분석 근거 테이블: <code>${item.evidence}</code></span>
        </div>
      </div>
    `;
  }

  // 1-4. Baseline Comparison Table in Accordion
  renderBaselineComparisonTable() {
    if (!this.bundle) return;
    const { baseline_performance } = this.bundle.tables;
    const tbody = document.getElementById('anl_baseline_tbody');
    if (!tbody) return;

    const row = (baseline_performance || []).find(
      r => r.target === this.state.target && r.split === this.state.split
    );

    if (!row) {
      tbody.innerHTML = `<tr><td colspan="7" class="td-center text-muted">선택한 분할 데이터가 없습니다.</td></tr>`;
      return;
    }

    const persDiff = row.persistence_rmse ? ((row.rmse_mean - row.persistence_rmse) / row.persistence_rmse * 100) : 0;
    const persDiffClass = persDiff <= 0 ? 'text-success font-bold' : 'text-danger font-bold';
    const persDiffText = persDiff <= 0 ? `${Math.abs(persDiff).toFixed(1)}% 개선` : `${persDiff.toFixed(1)}% 악화`;

    const r2Val = (row.r2_mean !== null && !isNaN(row.r2_mean)) ? row.r2_mean.toFixed(4) : '<span class="text-muted">미정의</span>';
    const cccVal = (row.ccc_mean !== null && !isNaN(row.ccc_mean)) ? row.ccc_mean.toFixed(4) : '-';

    tbody.innerHTML = `
      <tr class="row-highlight">
        <td><strong>선정 모델 (${(row.model || '').toUpperCase()})</strong></td>
        <td class="td-right font-bold">${row.rmse_mean.toFixed(4)}</td>
        <td class="td-right">${row.mae_mean.toFixed(4)}</td>
        <td class="td-right">${r2Val}</td>
        <td class="td-right">${cccVal}</td>
        <td class="td-right ${persDiffClass}">${persDiffText}</td>
        <td>검증 데이터에서 동결 확정된 최적 머신러닝 기준</td>
      </tr>
      <tr>
        <td>현재값 유지 (Persistence)</td>
        <td class="td-right">${row.persistence_rmse ? row.persistence_rmse.toFixed(4) : '-'}</td>
        <td class="td-right">${row.persistence_mae ? row.persistence_mae.toFixed(4) : '-'}</td>
        <td class="td-right text-muted">-</td>
        <td class="td-right text-muted">-</td>
        <td class="td-right text-muted">기준선 (0.0%)</td>
        <td>직전 조사일의 실제 수량을 그대로 예측값으로 사용하는 기준</td>
      </tr>
      <tr>
        <td>항상 0 예측 (Zero Predictor)</td>
        <td class="td-right">${row.zero_rmse ? row.zero_rmse.toFixed(4) : '-'}</td>
        <td class="td-right">${row.zero_mae ? row.zero_mae.toFixed(4) : '-'}</td>
        <td class="td-right text-muted">0.0000</td>
        <td class="td-right text-muted">0.0000</td>
        <td class="td-right text-muted">-</td>
        <td>정답 0 비율 착시 진단용 (실제 운영 대안으로 채택 불가)</td>
      </tr>
    `;
  }

  // 1-5. Crossed Strata Table in Accordion
  renderCrossedStrataTable() {
    if (!this.bundle) return;
    const { crossed_error_strata } = this.bundle.tables;
    const tbody = document.getElementById('anl_crossed_tbody');
    if (!tbody) return;

    const rows = (crossed_error_strata || []).filter(r => r.target === this.state.target);
    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="td-center text-muted">해당 타깃의 복합 교차 오차 데이터가 없습니다.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map(r => {
      const smallBadge = r.small_n ? `<span class="anl-badge-small-n">소표본</span>` : '';
      return `
        <tr>
          <td><span class="anl-tag-dim">${r.dimensions}</span></td>
          <td><strong>${r.level}</strong> ${smallBadge}</td>
          <td class="td-right">${r.n}건</td>
          <td class="td-right font-bold">${r.rmse_mean.toFixed(4)}</td>
        </tr>
      `;
    }).join('');
  }

  // =========================================================================
  // SUBTAB 2: AX DATA ROADMAP & HYPOTHESES
  // =========================================================================
  renderHypothesesMasterDetail() {
    if (!this.bundle) return;
    const { hypotheses } = this.bundle.tables;
    const listContainer = document.getElementById('anl_hyp_list');
    if (!listContainer || !hypotheses) return;

    // Render Left Master List
    listContainer.innerHTML = hypotheses.map(h => {
      const isSelected = h.id === this.state.selectedHypId;
      const prioClass = h.priority.toLowerCase();
      const isNewAx = h.category.includes('신규') || h.category.includes('행동') || h.category.includes('맥락') || h.category.includes('제약');
      const catBadge = isNewAx ? '<span class="anl-cat-badge new">신규 AX</span>' : '<span class="anl-cat-badge exist">기존 입력</span>';

      return `
        <div class="anl-hyp-item ${isSelected ? 'active' : ''}" data-hyp-id="${h.id}">
          <div class="anl-hyp-item-top">
            <span class="anl-prio-badge ${prioClass}">${h.priority}</span>
            <strong class="anl-hyp-item-id">${h.id}</strong>
            ${catBadge}
          </div>
          <div class="anl-hyp-item-title">${h.hypothesis}</div>
          <div class="anl-hyp-item-meta">
            <span>상태: ${h.status}</span>
          </div>
        </div>
      `;
    }).join('');

    // Item click handler
    listContainer.querySelectorAll('.anl-hyp-item').forEach(item => {
      item.addEventListener('click', () => {
        listContainer.querySelectorAll('.anl-hyp-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        this.state.selectedHypId = item.dataset.hypId;
        this.renderHypothesisDetail();
      });
    });

    this.renderHypothesisDetail();
  }

  renderHypothesisDetail() {
    if (!this.bundle) return;
    const { hypotheses } = this.bundle.tables;
    const detailPanel = document.getElementById('anl_hyp_detail_panel');
    if (!detailPanel) return;

    const item = (hypotheses || []).find(h => h.id === this.state.selectedHypId) || hypotheses[0];
    if (!item) return;

    const prioClass = item.priority.toLowerCase();
    const isNewAx = item.category.includes('신규') || item.category.includes('행동') || item.category.includes('맥락') || item.category.includes('제약');

    detailPanel.innerHTML = `
      <div class="anl-hyp-detail-wrap">
        <div class="anl-hyp-detail-header">
          <div class="anl-hyp-detail-badge-row">
            <span class="anl-prio-badge ${prioClass}">${item.priority}</span>
            <span class="anl-badge-tag">${item.category}</span>
            <span class="anl-badge-status">${item.status}</span>
          </div>
          <h3 class="anl-hyp-detail-title">[${item.id}] ${item.hypothesis}</h3>
        </div>

        <div class="anl-hyp-detail-sections">
          <!-- 1. 관측 근거 (확인된 사실) -->
          <div class="anl-hyp-box finding">
            <div class="anl-hyp-box-title">
              <span>🔍</span> <strong>현재 데이터에서 확인된 관측 사실 (Finding)</strong>
              <span class="anl-badge-confirmed">확인된 사실</span>
            </div>
            <p class="anl-hyp-box-text">${item.evidence}</p>
          </div>

          <!-- 2. 개선 가설 (미검증) -->
          <div class="anl-hyp-box hypothesis">
            <div class="anl-hyp-box-title">
              <span>💡</span> <strong>AX 데이터 개선 가설 (Proposed Hypothesis)</strong>
              <span class="anl-badge-unverified">미검증 가설</span>
            </div>
            <p class="anl-hyp-box-text">${item.hypothesis}</p>
          </div>

          <!-- 3. 후보 데이터 및 수집/연결 방법 -->
          <div class="anl-hyp-meta-grid">
            <div class="anl-hyp-meta-col">
              <span class="meta-col-lbl">📦 후보 데이터 항목 (${isNewAx ? '신규 수집 대상' : '기존 이력 활용'})</span>
              <div class="meta-col-val font-semibold">${item.candidate}</div>
            </div>
            <div class="anl-hyp-meta-col">
              <span class="meta-col-lbl">🛠️ 공정 검증 방법</span>
              <div class="meta-col-val">${item.validation}</div>
            </div>
            <div class="anl-hyp-meta-col full">
              <span class="meta-col-lbl">🎯 목표 검증 KPI</span>
              <div class="meta-col-val highlight font-bold">${item.kpi}</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // 2-2. Schema Explorer Table
  renderSchemaTable() {
    if (!this.bundle) return;
    const { ax_field_schema } = this.bundle.tables;
    const tbody = document.getElementById('anl_schema_tbody');
    const infoEl = document.getElementById('anl_schema_search_info');
    if (!tbody || !ax_field_schema) return;

    const filter = this.state.schemaTableFilter;
    const query = this.state.schemaSearchQuery;

    let rows = ax_field_schema;
    if (filter !== 'all') {
      rows = rows.filter(r => r.table === filter);
    }
    if (query) {
      rows = rows.filter(r => {
        return (r.field && r.field.toLowerCase().includes(query)) ||
               (r.table && r.table.toLowerCase().includes(query)) ||
               (r.source && r.source.toLowerCase().includes(query)) ||
               (r.hypothesis && r.hypothesis.toLowerCase().includes(query)) ||
               (r.join_key && r.join_key.toLowerCase().includes(query)) ||
               (r.unit && r.unit.toLowerCase().includes(query));
      });
    }

    if (infoEl) {
      infoEl.textContent = `총 88개 중 ${rows.length}개 표시 중`;
    }

    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="td-center text-muted">검색 조건에 일치하는 필드가 없습니다.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map(r => {
      const meta = SCHEMA_TABLE_META[r.table] || { label: r.table };
      const tableShort = meta.label.split(' ')[0];
      return `
        <tr>
          <td><span class="anl-tag-table">${tableShort}</span></td>
          <td><code>${r.field}</code></td>
          <td>${r.type} <span class="text-muted">(${r.unit})</span></td>
          <td>${r.source}</td>
          <td>${r.cadence}</td>
          <td><span class="text-muted font-mono" style="font-size:11px;">${r.join_key}</span></td>
          <td><span class="anl-tag-rule">${r.availability_rule}</span></td>
          <td><strong>${r.hypothesis}</strong></td>
        </tr>
      `;
    }).join('');
  }

  // =========================================================================
  // SUBTAB 3: VALIDATION PROTOCOL & KPI
  // =========================================================================
  renderComparisonArmsPipeline() {
    if (!this.bundle) return;
    const { comparison_arms } = this.bundle.tables;
    const container = document.getElementById('anl_arms_container');
    if (!container || !comparison_arms) return;

    container.innerHTML = comparison_arms.map(arm => {
      const isSelected = arm.arm === this.state.selectedArmId;
      const isAblation = arm.arm === 'ABL';
      const isToBe = arm.arm.startsWith('A');
      let typeClass = isAblation ? 'arm-abl' : (isToBe ? 'arm-to-be' : 'arm-as-is');
      let badgeText = isAblation ? '제거 검증' : (isToBe ? 'To-Be 미실험' : 'As-Is 기준');

      return `
        <div class="anl-arm-step-card ${typeClass} ${isSelected ? 'active' : ''}" data-arm-id="${arm.arm}">
          <div class="arm-step-header">
            <span class="arm-step-code">${arm.arm}</span>
            <span class="arm-step-badge">${badgeText}</span>
          </div>
          <div class="arm-step-name">${arm.name}</div>
          <div class="arm-step-sub">${arm.purpose}</div>
        </div>
      `;
    }).join('');

    container.querySelectorAll('.anl-arm-step-card').forEach(card => {
      card.addEventListener('click', () => {
        container.querySelectorAll('.anl-arm-step-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        this.state.selectedArmId = card.dataset.armId;
        this.renderArmDetail();
      });
    });

    this.renderArmDetail();
  }

  renderArmDetail() {
    if (!this.bundle) return;
    const { comparison_arms } = this.bundle.tables;
    const panel = document.getElementById('anl_arm_detail_panel');
    if (!panel) return;

    const arm = (comparison_arms || []).find(a => a.arm === this.state.selectedArmId) || comparison_arms[0];
    if (!arm) return;

    const isToBe = arm.arm.startsWith('A') || arm.arm === 'ABL';

    panel.innerHTML = `
      <div class="anl-arm-detail-box">
        <div class="arm-detail-header-row">
          <div class="arm-detail-title-wrap">
            <span class="anl-arm-lg-code">${arm.arm}</span>
            <h4 class="anl-arm-lg-title">${arm.name}</h4>
          </div>
          <span class="anl-status-badge ${isToBe ? 'unexp' : 'exp'}">
            ${isToBe ? '⚠️ 향후 신규 코호트 PoC 미실험 단계' : '✔️ 현재 완료된 실험군'}
          </span>
        </div>

        <div class="anl-arm-detail-grid">
          <div class="arm-detail-col">
            <span class="arm-col-lbl">📥 투입 입력변수군 구성</span>
            <div class="arm-col-val font-semibold">${arm.inputs}</div>
          </div>
          <div class="arm-detail-col">
            <span class="arm-col-lbl">🎯 검증 목적 및 효과</span>
            <div class="arm-col-val">${arm.purpose}</div>
          </div>
          <div class="arm-detail-col full">
            <span class="arm-col-lbl">⚖️ 공정 비교 대상 (Baseline)</span>
            <div class="arm-col-val highlight">${arm.comparison}</div>
          </div>
        </div>
      </div>
    `;
  }

  // 3-2. Target-Specific KPI Card
  renderTargetKpiCard() {
    if (!this.bundle) return;
    const { kpi_planning_reference } = this.bundle.tables;
    const container = document.getElementById('anl_target_kpi_grid');
    if (!container || !kpi_planning_reference) return;

    const item = kpi_planning_reference.find(r => r.target === this.state.target) || kpi_planning_reference[0];
    if (!item) return;

    container.innerHTML = `
      <div class="anl-kpi-item-card">
        <span class="kpi-item-lbl">현 기준 RMSE (As-Is)</span>
        <div class="kpi-item-val font-bold">${item.reference_rmse.toFixed(4)}</div>
        <span class="kpi-item-sub">현재 동결 선정 모델</span>
      </div>
      <div class="anl-kpi-item-card highlight">
        <span class="kpi-item-lbl">10% RMSE 개선 기획 예시</span>
        <div class="kpi-item-val font-bold text-primary">${item.illustrative_10pct_rmse.toFixed(4)}</div>
        <span class="kpi-item-sub text-primary">목표 오차 상한 제안</span>
      </div>
      <div class="anl-kpi-item-card">
        <span class="kpi-item-lbl">현 수량변화 MAE</span>
        <div class="kpi-item-val font-bold">${item.reference_change_mae.toFixed(4)}</div>
        <span class="kpi-item-sub">수량 변동 구간 절대오차</span>
      </div>
      <div class="anl-kpi-item-card highlight">
        <span class="kpi-item-lbl">15% 변화 MAE 개선 기획 예시</span>
        <div class="kpi-item-val font-bold text-primary">${item.illustrative_15pct_change_mae.toFixed(4)}</div>
        <span class="kpi-item-sub text-primary">목표 변동 오차 제안</span>
      </div>
    `;
  }

  // 3-3. Full KPI Planning Reference Table in Accordion
  renderFullKpiTable() {
    if (!this.bundle) return;
    const { kpi_planning_reference } = this.bundle.tables;
    const tbody = document.getElementById('anl_kpi_tbody');
    if (!tbody || !kpi_planning_reference) return;

    tbody.innerHTML = kpi_planning_reference.map(r => {
      const meta = TARGET_META[r.target] || { name: r.target, icon: '🌱' };
      const isCurrentTarget = r.target === this.state.target;
      return `
        <tr class="${isCurrentTarget ? 'row-highlight' : ''}">
          <td><strong>${meta.icon} ${meta.name}</strong></td>
          <td class="td-right font-bold">${r.reference_rmse.toFixed(4)}</td>
          <td class="td-right text-primary font-bold">${r.illustrative_10pct_rmse.toFixed(4)}</td>
          <td class="td-right">${r.reference_change_mae.toFixed(4)}</td>
          <td class="td-right text-primary font-bold">${r.illustrative_15pct_change_mae.toFixed(4)}</td>
          <td class="td-right text-muted">${r.reference_persistence_rmse.toFixed(4)}</td>
          <td class="td-right text-muted">${r.reference_zero_rmse.toFixed(4)}</td>
          <td><span class="anl-badge-tag warning">기획 참고치</span></td>
        </tr>
      `;
    }).join('');
  }
}
