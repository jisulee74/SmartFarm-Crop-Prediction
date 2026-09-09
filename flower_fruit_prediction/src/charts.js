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

export class ChartManager {
  constructor() {
    this.chartInstances = new Map();
    this.initResizeListener();
  }

  initResizeListener() {
    window.addEventListener('resize', () => {
      this.resizeAll();
    });
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

    // 시간축 시리즈 데이터: [ [Date, Value, validCount, min, max], ... ]
    const seriesData = trendData.map(item => [
      item.date,
      item.value,
      item.validCount,
      item.min,
      item.max,
      item.totalEntitiesToday
    ]);

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
          type: 'cross',
          crossStyle: { color: '#94a3b8' },
          lineStyle: { color: config.color, width: 1.5, type: 'dashed' }
        },
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: config.color,
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
              <span style="font-size: 15px; font-weight: 800; color: ${config.color}">${avgVal} ${config.unit}</span>
            </div>
            <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 2px;">
              👥 <strong>유효 관측 개체 수:</strong> <span style="color:#fde047; font-weight:700">${validCount}개체</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8;">
              📈 당일 관측 범위: ${minVal} ~ ${maxVal} ${config.unit}
            </div>
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
          symbol: 'circle',
          symbolSize: 6,
          itemStyle: {
            color: config.color,
            borderColor: '#ffffff',
            borderWidth: 1.5
          },
          lineStyle: {
            width: 2.5,
            color: config.color
          },
          areaStyle: {
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
        symbol: 'circle',
        symbolSize: 7,
        itemStyle: {
          color: color,
          borderColor: '#ffffff',
          borderWidth: 1.5
        },
        lineStyle: {
          width: 2,
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
  }
}
