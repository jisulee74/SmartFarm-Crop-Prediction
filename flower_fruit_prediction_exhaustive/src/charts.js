/**
 * ECharts 기반 반응형 차트 렌더링 모듈
 * 8개 독립 차트 (토마토 4개, 딸기 4개) 구성 및 동적 업데이트
 */

// 지표별 메타데이터 정의
export const METRIC_CONFIGS = {
  // 토마토 4종
  tomato_flower_truss1: {
    id: 'chart_tomato_flower_truss1',
    crop: 'tomato',
    title: '토마토 1화방 꽃 수',
    subtitle: '개체별 1화방의 개화 관측 추이',
    unit: '개',
    color: '#FF6B6B',
    accentColor: 'rgba(255, 107, 107, 0.2)'
  },
  tomato_flower_truss2: {
    id: 'chart_tomato_flower_truss2',
    crop: 'tomato',
    title: '토마토 2화방 꽃 수',
    subtitle: '개체별 2화방의 개화 관측 추이',
    unit: '개',
    color: '#FF8E53',
    accentColor: 'rgba(255, 142, 83, 0.2)'
  },
  tomato_flower_truss3: {
    id: 'chart_tomato_flower_truss3',
    crop: 'tomato',
    title: '토마토 3화방 꽃 수',
    subtitle: '개체별 3화방의 개화 관측 추이',
    unit: '개',
    color: '#FFA07A',
    accentColor: 'rgba(255, 160, 122, 0.2)'
  },
  tomato_flower_total: {
    id: 'chart_tomato_flower_total',
    crop: 'tomato',
    title: '토마토 전체 꽃 수',
    subtitle: '개체별 1·2·3화방 개화 관측 합계 추이',
    unit: '개',
    color: '#E03131',
    accentColor: 'rgba(224, 49, 49, 0.25)'
  },

  // 딸기 4종
  strawberry_fruit_truss1: {
    id: 'chart_strawberry_fruit_truss1',
    crop: 'strawberry',
    title: '딸기 1화방 착과수',
    subtitle: '개체별 1화방의 착과 관측 추이',
    unit: '개',
    color: '#F06595',
    accentColor: 'rgba(240, 101, 149, 0.2)'
  },
  strawberry_fruit_truss2: {
    id: 'chart_strawberry_fruit_truss2',
    crop: 'strawberry',
    title: '딸기 2화방 착과수',
    subtitle: '개체별 2화방의 착과 관측 추이',
    unit: '개',
    color: '#CC5DE8',
    accentColor: 'rgba(204, 93, 232, 0.2)'
  },
  strawberry_fruit_truss3: {
    id: 'chart_strawberry_fruit_truss3',
    crop: 'strawberry',
    title: '딸기 3화방 착과수',
    subtitle: '개체별 3화방의 착과 관측 추이',
    unit: '개',
    color: '#845EF7',
    accentColor: 'rgba(132, 94, 247, 0.2)'
  },
  strawberry_fruit_total: {
    id: 'chart_strawberry_fruit_total',
    crop: 'strawberry',
    title: '딸기 전체 착과수',
    subtitle: '개체별 1·2·3화방 착과 관측 합계 추이',
    unit: '개',
    color: '#D6336C',
    accentColor: 'rgba(214, 51, 108, 0.25)'
  }
};

import { fetchAnalysisBundle } from './analysis_api.js';

export class ChartManager {
  constructor() {
    this.chartInstances = new Map();
    this.qualityData = null;
    this.initResizeListener();
    this.loadQualityData();
    this.initDropdownListeners();
  }

  initResizeListener() {
    window.addEventListener('resize', () => {
      this.resizeAll();
    });
  }

  async loadQualityData() {
    try {
      const bundle = await fetchAnalysisBundle();
      if (bundle && bundle.tables && bundle.tables.crop_cycle_quality) {
        this.qualityData = bundle.tables.crop_cycle_quality;
      }
    } catch (e) {
      console.warn('[ChartManager] Failed to load quality data:', e);
    }
  }

  initDropdownListeners() {
    ['tomato', 'strawberry'].forEach(crop => {
      const facEl = document.getElementById(`${crop}_facility_select`);
      const cropEl = document.getElementById(`${crop}_crop_select`);
      const resetBtn = document.getElementById(`${crop}_btn_reset`);
      if (facEl) facEl.addEventListener('change', () => setTimeout(() => this.updateUsabilityCard(crop), 50));
      if (cropEl) cropEl.addEventListener('change', () => setTimeout(() => this.updateUsabilityCard(crop), 50));
      if (resetBtn) resetBtn.addEventListener('click', () => setTimeout(() => this.updateUsabilityCard(crop), 100));
    });
  }

  updateUsabilityCard(cropKey) {
    const card = document.getElementById(`${cropKey}_usability_card`);
    if (!card) return;

    const facEl = document.getElementById(`${cropKey}_facility_select`);
    const cropEl = document.getElementById(`${cropKey}_crop_select`);
    const selectedFac = facEl ? facEl.value : 'ALL';
    const selectedCrop = cropEl ? cropEl.value : 'ALL';

    if (selectedCrop === 'ALL') {
      card.style.display = 'none';
      return;
    }

    const item = (this.qualityData || []).find(r => 
      r.crop === cropKey && 
      String(r.crop_sn) === String(selectedCrop) &&
      (selectedFac === 'ALL' || r.facility_id === selectedFac)
    );

    if (!item) {
      card.style.display = 'none';
      return;
    }

    card.style.display = 'block';
    const isSingle = Boolean(item.is_single_observation);
    card.className = `prediction-usability-card ${isSingle ? 'state-warning' : 'state-normal'}`;

    const titleIcon = isSingle ? '⚠️' : '✅';
    const titleText = isSingle
      ? '작기 품질 및 예측 활용 가능성 진단: 적격성 경고 (예측쌍 생성 불가)'
      : '작기 품질 및 예측 활용 가능성 진단: 정상 적격';
    const badgeHtml = isSingle
      ? '<span class="usability-state-badge badge-warn">⚠️ 단일 관측 시점 (모델링 코호트 제외)</span>'
      : '<span class="usability-state-badge badge-ok">✅ 예측 적격 작기 (연속 관측)</span>';

    const pairsSub = item.prediction_pairs === 0 ? 'shift(-1) 쌍 생성 불가' : '유효 지도학습 쌍';
    const noticeClass = isSingle ? 'notice-warn' : 'notice-info';
    const noticeIcon = isSingle ? '⚠️' : 'ℹ️';

    let noticeText = item.custom_notice;
    if (!noticeText) {
      noticeText = isSingle
        ? `이 작기는 실제 관측일이 ${item.unique_dates}회(${item.dates_str})에 불과하여 다음 조사값 예측쌍을 생성할 수 없습니다. 원본 관측은 보존되지만 모델 학습·검증·평가에서는 제외됩니다.`
        : `이 작기는 ${item.unique_dates}회에 걸쳐 총 ${item.actual_rows}건의 조사가 이루어졌으며, ${item.prediction_pairs}개의 유효한 다음 조사 예측쌍이 생성되어 모델링에 정상 활용됩니다.`;
    }

    card.innerHTML = `
      <div class="usability-card-header">
        <div class="usability-card-title">
          <span>${titleIcon}</span>
          <span>${titleText}</span>
        </div>
        ${badgeHtml}
      </div>
      <div class="usability-stats-grid">
        <div class="usability-stat-item">
          <span class="usability-stat-label">실제 관측 행 수</span>
          <span class="usability-stat-val">${Number(item.actual_rows).toLocaleString()}건</span>
          <span class="usability-stat-sub">실제 기록된 관측 행</span>
        </div>
        <div class="usability-stat-item ${isSingle ? 'highlight-warn' : ''}">
          <span class="usability-stat-label">관측된 고유 날짜 수</span>
          <span class="usability-stat-val">${item.unique_dates}일</span>
          <span class="usability-stat-sub">${item.dates_str}</span>
        </div>
        <div class="usability-stat-item">
          <span class="usability-stat-label">관측 개체 수</span>
          <span class="usability-stat-val">${item.sample_count}개체</span>
          <span class="usability-stat-sub">${item.samples_str}</span>
        </div>
        <div class="usability-stat-item ${item.prediction_pairs === 0 ? 'highlight-warn' : ''}">
          <span class="usability-stat-label">다음 관측 예측쌍 수</span>
          <span class="usability-stat-val">${item.prediction_pairs}쌍</span>
          <span class="usability-stat-sub">${pairsSub}</span>
        </div>
        <div class="usability-stat-item ${item.excluded_rows > 0 ? 'highlight-warn' : ''}">
          <span class="usability-stat-label">제외 행 수 및 사유</span>
          <span class="usability-stat-val">${item.excluded_rows}건</span>
          <span class="usability-stat-sub">${item.exclusion_reasons || '제외 사유 없음'}</span>
        </div>
      </div>
      <div class="usability-notice-box ${noticeClass}">
        <span style="font-size: 16px;">${noticeIcon}</span>
        <span><strong>품질 판정:</strong> ${noticeText}</span>
      </div>
    `;
  }

  resizeAll() {
    for (const chart of this.chartInstances.values()) {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    }
  }

  getOrCreateChart(domId) {
    const dom = document.getElementById(domId);
    if (!dom) return null;

    let chart = this.chartInstances.get(domId);
    if (!chart || chart.isDisposed()) {
      chart = window.echarts.init(dom, null, { renderer: 'canvas' });
      this.chartInstances.set(domId, chart);
    }
    return chart;
  }

  /**
   * 개체당 평균 모드 차트 렌더링
   */
  renderAverageChart(metricKey, trendData) {
    const config = METRIC_CONFIGS[metricKey];
    if (!config) return;

    const chart = this.getOrCreateChart(config.id);
    if (!chart) return;

    const container = document.getElementById(config.id);
    const emptyOverlay = container.parentElement.querySelector('.chart-empty-state');

    if (!trendData || trendData.length === 0) {
      chart.clear();
      if (emptyOverlay) emptyOverlay.style.display = 'flex';
      return;
    }
    if (emptyOverlay) emptyOverlay.style.display = 'none';

    const isSingleDate = (trendData.length === 1);
    const cardBadge = document.getElementById(`badge_${metricKey}`);
    if (cardBadge) {
      if (isSingleDate) {
        cardBadge.textContent = '단일 관측 시점';
        cardBadge.className = 'chart-mode-badge badge-single-obs';
      } else {
        cardBadge.textContent = '개체당 평균';
        cardBadge.className = 'chart-mode-badge';
        cardBadge.style.background = '#eff6ff';
        cardBadge.style.color = '#2563eb';
        cardBadge.style.borderColor = '#bfdbfe';
      }
    }

    // 시간축 시리즈 데이터: [ [Date, Value, validCount, min, max], ... ]
    const seriesData = trendData.map(item => [
      item.date,
      item.value,
      item.validCount,
      item.min,
      item.max,
      item.totalEntitiesToday
    ]);

    const singleAlertHtml = isSingleDate ? `
      <div style="margin-top: 6px; padding: 4px 8px; background: rgba(217, 119, 6, 0.25); border-left: 3px solid #f59e0b; color: #fde047; font-size: 11px; font-weight: 700;">
        ⚠️ 단일 관측 시점: 다음 조사 예측쌍 생성 불가 (모델링 제외)
      </div>` : '';

    const option = {
      backgroundColor: 'transparent',
      grid: {
        left: 48,
        right: 18,
        top: 32,
        bottom: 48,
        containLabel: false
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: isSingleDate ? 'none' : 'cross',
          crossStyle: { color: '#94a3b8' },
          lineStyle: { color: config.color, width: 1.5, type: 'dashed' }
        },
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: isSingleDate ? '#f59e0b' : config.color,
        borderWidth: 1.5,
        padding: [10, 14],
        textStyle: { color: '#ffffff', fontSize: 13 },
        formatter: (params) => {
          if (!params || params.length === 0) return '';
          const pt = params[0];
          const data = pt.data;
          if (!data) return '';
          const [dateStr, avgVal, validCount, minVal, maxVal] = data;

          return `
            <div style="font-weight: 700; margin-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 4px;">
              📅 조사일: <span style="color:#60a5fa">${dateStr}</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 4px;">
              <span>📊 <strong>개체당 평균:</strong></span>
              <span style="font-size: 15px; font-weight: 800; color: ${isSingleDate ? '#f59e0b' : config.color}">${avgVal} ${config.unit}</span>
            </div>
            <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 2px;">
              👥 <strong>유효 관측 개체 수:</strong> <span style="color:#fde047; font-weight:700">${validCount}개체</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8;">
              📈 당일 관측 범위: ${minVal} ~ ${maxVal} ${config.unit}
            </div>
            ${singleAlertHtml}
          `;
        }
      },
      toolbox: {
        feature: {
          dataZoom: { yAxisIndex: 'none', title: { zoom: '영역 확대', back: '확대 취소' } },
          restore: { title: '초기화' },
          saveAsImage: { title: '이미지 저장', name: `${config.title}_평균추이` }
        },
        right: 10,
        top: 2,
        iconStyle: { borderColor: '#64748b' }
      },
      xAxis: {
        type: 'time',
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#cbd5e1' } },
        axisLabel: {
          color: '#64748b',
          fontSize: 11,
          formatter: '{yyyy}-{MM}-{dd}'
        },
        splitLine: {
          show: true,
          lineStyle: { color: '#f1f5f9', type: 'dashed' }
        }
      },
      yAxis: {
        type: 'value',
        name: `개체당 평균 (${config.unit})`,
        nameTextStyle: { color: '#64748b', fontSize: 11, padding: [0, 0, 0, 10] },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#64748b', fontSize: 11 },
        splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          type: 'slider',
          show: true,
          height: 18,
          bottom: 10,
          borderColor: 'transparent',
          backgroundColor: '#f1f5f9',
          fillerColor: config.accentColor,
          handleStyle: { color: config.color, borderColor: config.color },
          textStyle: { color: '#94a3b8', fontSize: 10 }
        }
      ],
      series: [
        {
          name: '개체당 평균',
          type: 'line',
          smooth: false,
          showSymbol: true,
          symbol: isSingleDate ? 'diamond' : 'circle',
          symbolSize: isSingleDate ? 12 : 6,
          itemStyle: {
            color: isSingleDate ? '#d97706' : config.color,
            borderColor: '#ffffff',
            borderWidth: isSingleDate ? 2 : 1.5
          },
          lineStyle: {
            width: isSingleDate ? 0 : 2.5,
            color: config.color
          },
          areaStyle: isSingleDate ? null : {
            color: new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: config.accentColor },
              { offset: 1, color: 'rgba(255, 255, 255, 0.01)' }
            ])
          },
          data: seriesData
        }
      ]
    };

    chart.setOption(option, true);
    this.updateUsabilityCard(config.crop);
  }

  /**
   * 개체별 추이 모드 차트 렌더링
   * - 시설이나 작기가 다른 점은 연결되지 않도록 각각 개별 시리즈로 분리
   * - 실제 관측값 그대로 표시 (평균 미사용)
   */
  renderIndividualChart(metricKey, individualGroups) {
    const config = METRIC_CONFIGS[metricKey];
    if (!config) return;

    const chart = this.getOrCreateChart(config.id);
    if (!chart) return;

    const container = document.getElementById(config.id);
    const emptyOverlay = container.parentElement.querySelector('.chart-empty-state');

    if (!individualGroups || individualGroups.length === 0) {
      chart.clear();
      if (emptyOverlay) emptyOverlay.style.display = 'flex';
      return;
    }
    if (emptyOverlay) emptyOverlay.style.display = 'none';

    const isSingleObservationGroup = (individualGroups.length > 0 && individualGroups.every(g => g.points.length <= 1));
    const cardBadge = document.getElementById(`badge_${metricKey}`);
    if (cardBadge) {
      if (isSingleObservationGroup) {
        cardBadge.textContent = `단일 관측 시점 (${individualGroups.length}개 개체)`;
        cardBadge.className = 'chart-mode-badge badge-single-obs';
      } else {
        cardBadge.textContent = `개체별 실제값 (${individualGroups.length}개 계열)`;
        cardBadge.className = 'chart-mode-badge';
        cardBadge.style.background = '#f3e8ff';
        cardBadge.style.color = '#7e22ce';
        cardBadge.style.borderColor = '#d8b4fe';
      }
    }

    // 색상 팔레트
    const palette = [
      '#e03131', '#2f9e44', '#1971c2', '#f08c00', '#9c36b5',
      '#0c8599', '#d6336c', '#4263eb', '#ae3ec9', '#7048e8'
    ];

    const series = individualGroups.map((group, idx) => {
      const color = palette[idx % palette.length];
      const seriesData = group.points.map(pt => [
        pt.date,
        pt.value,
        pt.facility_id,
        pt.crop_sn,
        pt.sample_num,
        pt.crop_start_date,
        pt.crop_end_date
      ]);

      return {
        name: group.entityName,
        type: 'line',
        smooth: false,
        showSymbol: true,
        symbol: isSingleObservationGroup ? 'diamond' : 'circle',
        symbolSize: isSingleObservationGroup ? 10 : 7,
        itemStyle: {
          color: color,
          borderColor: '#ffffff',
          borderWidth: 1.5
        },
        lineStyle: {
          width: isSingleObservationGroup ? 0 : 2,
          color: color
        },
        data: seriesData
      };
    });

    const option = {
      backgroundColor: 'transparent',
      toolbox: {
        feature: {
          dataZoom: { yAxisIndex: 'none', title: { zoom: '영역 확대', back: '확대 취소' } },
          restore: { title: '초기화' },
          saveAsImage: { title: '이미지 저장', name: `${config.title}_개체별추이` }
        },
        right: 10,
        top: 2,
        iconStyle: { borderColor: '#64748b' }
      },
      grid: {
        left: 48,
        right: 18,
        top: 36,
        bottom: 74,
        containLabel: false
      },
      legend: {
        type: 'scroll',
        orient: 'horizontal',
        bottom: 2,
        left: 'center',
        width: '92%',
        itemGap: 14,
        textStyle: { fontSize: 11, color: '#475569' },
        pageTextStyle: { color: '#64748b' },
        pageIconColor: '#2563eb',
        pageIconInactiveColor: '#cbd5e1'
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: '#38bdf8',
        borderWidth: 1.5,
        padding: [10, 14],
        textStyle: { color: '#ffffff', fontSize: 13 },
        formatter: (param) => {
          const [dateStr, val, facility, cropSn, sampleNum, startDate, endDate] = param.data;
          const cropPeriodStr = (startDate && endDate) ? `${startDate} ~ ${endDate}` : (cropSn || '-');
          return `
            <div style="font-weight: 700; margin-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 4px;">
              📅 조사일: <span style="color:#60a5fa">${dateStr}</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 6px;">
              <span>📍 <strong>실제 관측값:</strong></span>
              <span style="font-size: 16px; font-weight: 800; color: #38bdf8">${val} ${config.unit}</span>
            </div>
            <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 2px;">
              🏢 <strong>시설:</strong> ${facility}
            </div>
            <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 2px;">
              ⏱️ <strong>작기 기간:</strong> ${cropPeriodStr}
            </div>
            <div style="font-size: 12px; color: #fde047; font-weight: 700;">
              🏷️ <strong>개체 번호:</strong> #${sampleNum}
            </div>
          `;
        }
      },
      xAxis: {
        type: 'time',
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#cbd5e1' } },
        axisLabel: {
          color: '#64748b',
          fontSize: 11,
          formatter: '{yyyy}-{MM}-{dd}'
        },
        splitLine: {
          show: true,
          lineStyle: { color: '#f1f5f9', type: 'dashed' }
        }
      },
      yAxis: {
        type: 'value',
        name: `실제 관측값 (${config.unit})`,
        nameTextStyle: { color: '#64748b', fontSize: 11, padding: [0, 0, 8, 0] },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#64748b', fontSize: 11 },
        splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          type: 'slider',
          show: true,
          height: 16,
          bottom: 28,
          borderColor: 'transparent',
          backgroundColor: '#f1f5f9',
          fillerColor: config.accentColor,
          handleStyle: { color: config.color, borderColor: config.color },
          textStyle: { color: '#94a3b8', fontSize: 10 }
        }
      ],
      series: series
    };

    chart.setOption(option, true);
    this.updateUsabilityCard(config.crop);
  }
}
