/**
 * 메인 애플리케이션 컨트롤러
 * 데이터 로딩, 상태 관리, 이벤트 바인딩, 차트 동기화
 */

import { parseCSV } from './parser.js';
import {
  computeSummaryStats,
  getFilterOptions,
  filterRows,
  aggregateAverageTrend,
  extractIndividualTrends
} from './analytics.js';
import { ChartManager, METRIC_CONFIGS } from './charts.js';
import { ExperimentViewController } from './experiment_view.js';
import { AnalysisViewController } from './analysis_view.js';

// 전역 상태
const state = {
  tomato: {
    rawRows: [],
    filteredRows: [],
    mode: 'average', // 'average' | 'individual'
    filters: {
      facility_id: 'ALL',
      crop_sn: 'ALL',
      sample_num: 'ALL',
      startDate: '',
      endDate: ''
    },
    metrics: [
      'tomato_flower_truss1',
      'tomato_flower_truss2',
      'tomato_flower_truss3',
      'tomato_flower_total'
    ]
  },
  strawberry: {
    rawRows: [],
    filteredRows: [],
    mode: 'average', // 'average' | 'individual'
    filters: {
      facility_id: 'ALL',
      crop_sn: 'ALL',
      sample_num: 'ALL',
      startDate: '',
      endDate: ''
    },
    metrics: [
      'strawberry_fruit_truss1',
      'strawberry_fruit_truss2',
      'strawberry_fruit_truss3',
      'strawberry_fruit_total'
    ]
  }
};

let chartManager = null;
let expViewController = null;
let anlViewController = null;

// 토스트 메시지 표시
function showToast(message, duration = 3500) {
  const toast = document.getElementById('alert_toast');
  if (!toast) return;
  toast.textContent = message;
  toast.style.display = 'flex';
  setTimeout(() => {
    toast.style.display = 'none';
  }, duration);
}

// 애플리케이션 초기화
async function initApp() {
  try {
    chartManager = new ChartManager();
    expViewController = new ExperimentViewController();
    anlViewController = new AnalysisViewController();

    // 1. 주 네비게이션 탭 (관측 현황 vs 실험 결과 비교 vs 생육 예측 분석)
    initPrimaryTabs();

    // 2. 네비게이션 탭 이벤트 설정 (관측 뷰 전체/토마토/딸기)
    initNavTabs();

    // 3. CSV 파일 로드 및 파싱
    await loadDatasets();

    // 4. 필터 UI 및 이벤트 바인딩
    setupFilters('tomato');
    setupFilters('strawberry');

    // 5. 초기 차트 렌더링
    updateCropView('tomato');
    updateCropView('strawberry');

  } catch (error) {
    console.error('[Dashboard Error]', error);
    showToast(`⚠️ 오류 발생: ${error.message}`, 6000);
  }
}

// 주 네비게이션 탭 바인딩 (관측 현황 vs 실험 결과 비교 vs 생육 예측 분석)
function initPrimaryTabs() {
  const primaryTabs = document.querySelectorAll('.primary-tab-btn');
  primaryTabs.forEach(btn => {
    btn.addEventListener('click', async () => {
      primaryTabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const targetTabId = btn.dataset.tab;
      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
      });

      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      // 실험 뷰 탭 활성화 시 초기화 및 차트 리사이즈
      if (targetTabId === 'tab-experiments') {
        if (!expViewController.initialized) {
          await expViewController.init();
        } else if (expViewController.chartInstance) {
          setTimeout(() => expViewController.chartInstance.resize(), 100);
        }
      } else if (targetTabId === 'tab-analysis') {
        // 생육 예측 분석 탭 활성화 시 초기화
        if (!anlViewController.initialized) {
          await anlViewController.init();
        }
      } else if (targetTabId === 'tab-observation') {
        setTimeout(() => chartManager.resizeAll(), 100);
      }
    });
  });
}

// 상단 네비게이션 탭 바인딩
function initNavTabs() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const target = tab.dataset.target;
      const tomatoSec = document.getElementById('tomato-section');
      const strawberrySec = document.getElementById('strawberry-section');

      if (target === 'all') {
        tomatoSec.style.display = 'block';
        strawberrySec.style.display = 'block';
      } else if (target === 'tomato-section') {
        tomatoSec.style.display = 'block';
        strawberrySec.style.display = 'none';
      } else if (target === 'strawberry-section') {
        tomatoSec.style.display = 'none';
        strawberrySec.style.display = 'block';
      }

      setTimeout(() => chartManager.resizeAll(), 100);
    });
  });
}

// CSV 데이터 로드
async function loadDatasets() {
  // 토마토 로드
  try {
    const tomatoRes = await fetch('./tomato_observed_counts_by_crop_cycle.csv');
    if (!tomatoRes.ok) throw new Error(`토마토 CSV 파일을 불러올 수 없습니다. (상태 코드: ${tomatoRes.status})`);
    const tomatoText = await tomatoRes.text();
    const parsedTomato = parseCSV(tomatoText);
    state.tomato.rawRows = parsedTomato.rows;

    if (parsedTomato.duplicateWarnings.length > 0) {
      console.warn('토마토 중복 레코드 경고:', parsedTomato.duplicateWarnings);
      showToast(`⚠️ 토마토 데이터에서 ${parsedTomato.duplicateWarnings.length}건의 중복 식별자 키가 발견되었습니다.`, 5000);
    }

    renderSummaryCards('tomato', parsedTomato.rows);
  } catch (err) {
    console.error('토마토 데이터 로드 실패:', err);
    document.getElementById('tomato_summary_badge').textContent = '로드 실패';
    throw err;
  }

  // 딸기 로드
  try {
    const strawberryRes = await fetch('./strawberry_observed_counts_by_crop_cycle.csv');
    if (!strawberryRes.ok) throw new Error(`딸기 CSV 파일을 불러올 수 없습니다. (상태 코드: ${strawberryRes.status})`);
    const strawberryText = await strawberryRes.text();
    const parsedStrawberry = parseCSV(strawberryText);
    state.strawberry.rawRows = parsedStrawberry.rows;

    if (parsedStrawberry.duplicateWarnings.length > 0) {
      console.warn('딸기 중복 레코드 경고:', parsedStrawberry.duplicateWarnings);
      showToast(`⚠️ 딸기 데이터에서 ${parsedStrawberry.duplicateWarnings.length}건의 중복 식별자 키가 발견되었습니다.`, 5000);
    }

    renderSummaryCards('strawberry', parsedStrawberry.rows);
  } catch (err) {
    console.error('딸기 데이터 로드 실패:', err);
    document.getElementById('strawberry_summary_badge').textContent = '로드 실패';
    throw err;
  }
}

// 상단 요약 카드 렌더링
function renderSummaryCards(cropKey, rows) {
  const stats = computeSummaryStats(rows);

  const badgeEl = document.getElementById(`${cropKey}_summary_badge`);
  const periodEl = document.getElementById(`${cropKey}_period`);
  const facilityEl = document.getElementById(`${cropKey}_facility_count`);
  const cropPerFacEl = document.getElementById(`${cropKey}_crop_per_facility`);
  const entityEl = document.getElementById(`${cropKey}_entity_count`);
  const totalRowsEl = document.getElementById(`${cropKey}_total_rows`);

  if (badgeEl) badgeEl.textContent = '정상 로드 완료';
  if (periodEl) periodEl.textContent = `${stats.periodStart} ~ ${stats.periodEnd}`;
  if (facilityEl) facilityEl.textContent = `${stats.facilityCount}개 시설`;
  if (cropPerFacEl) {
    cropPerFacEl.textContent = `시설별 작기 수: 평균 ${stats.cropCyclePerFacility.avg}개 (총 ${stats.cropCycleCount}작기)`;
  }
  if (entityEl) entityEl.textContent = `${stats.uniqueEntityCount.toLocaleString()}개 개체`;
  if (totalRowsEl) totalRowsEl.textContent = `${stats.totalObservations.toLocaleString()}건`;
}

// 필터 컨트롤 및 이벤트 설정
function setupFilters(cropKey) {
  const cropState = state[cropKey];
  const { rawRows } = cropState;

  const facilitySelect = document.getElementById(`${cropKey}_facility_select`);
  const cropSelect = document.getElementById(`${cropKey}_crop_select`);
  const sampleSelect = document.getElementById(`${cropKey}_sample_select`);
  const startDateInput = document.getElementById(`${cropKey}_start_date`);
  const endDateInput = document.getElementById(`${cropKey}_end_date`);
  const resetBtn = document.getElementById(`${cropKey}_btn_reset`);
  const modeGroup = document.getElementById(`${cropKey}_mode_toggle`);

  const sampleHint = document.getElementById(`${cropKey}_sample_hint`);

  // 개체번호 컨트롤 상태 동기화 (개체당 평균 모드 vs 개체별 상세 모드)
  function syncSampleControlState() {
    if (cropState.mode === 'average') {
      cropState.filters.sample_num = 'ALL';
      sampleSelect.value = 'ALL';
      sampleSelect.disabled = true;
      if (sampleHint) sampleHint.style.display = 'inline-block';
    } else {
      sampleSelect.disabled = false;
      if (sampleHint) sampleHint.style.display = 'none';
    }
  }

  // 드롭다운 옵션 갱신
  function updateDropdownOptions() {
    const { facilities, cropSns, allCropSns, sampleNums } = getFilterOptions(rawRows, cropState.filters);

    // 1. 시설 옵션
    const currentFac = cropState.filters.facility_id;
    facilitySelect.innerHTML = '<option value="ALL">전체 시설</option>';
    facilities.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      if (f === currentFac) opt.selected = true;
      facilitySelect.appendChild(opt);
    });

    // 2. 작기 옵션 (시설이 ALL일 때는 전체 18개/28개 작기를 시설별 그룹으로 모두 표시)
    const currentCrop = cropState.filters.crop_sn;
    cropSelect.innerHTML = '<option value="ALL">전체 작기 (기간)</option>';

    if (cropState.filters.facility_id === 'ALL') {
      const facilityGroups = new Map();
      allCropSns.forEach(c => {
        if (!facilityGroups.has(c.facility_id)) {
          facilityGroups.set(c.facility_id, []);
        }
        facilityGroups.get(c.facility_id).push(c);
      });

      const sortedFacs = Array.from(facilityGroups.keys()).sort();
      sortedFacs.forEach(facId => {
        const cropsInFac = facilityGroups.get(facId);
        const groupEl = document.createElement('optgroup');
        groupEl.label = `🏢 ${facId} (${cropsInFac.length}개 작기)`;

        cropsInFac.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.value;
          opt.textContent = `${c.label}`;
          if (c.value === currentCrop) opt.selected = true;
          groupEl.appendChild(opt);
        });

        cropSelect.appendChild(groupEl);
      });
    } else {
      // 특정 시설이 선택된 경우 해당 시설의 작기들만 표시
      cropSns.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.value;
        opt.textContent = c.label;
        if (c.value === currentCrop) opt.selected = true;
        cropSelect.appendChild(opt);
      });
    }

    // 3. 개체번호 옵션
    const currentSample = cropState.filters.sample_num;
    sampleSelect.innerHTML = '<option value="ALL">전체 개체</option>';
    sampleNums.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = `#${s} 개체`;
      if (s === currentSample) opt.selected = true;
      sampleSelect.appendChild(opt);
    });

    syncSampleControlState();
  }

  updateDropdownOptions();
  syncSampleControlState();

  // 시설 변경 이벤트
  facilitySelect.addEventListener('change', (e) => {
    cropState.filters.facility_id = e.target.value;
    cropState.filters.crop_sn = 'ALL';
    cropState.filters.sample_num = 'ALL';
    updateDropdownOptions();
    updateCropView(cropKey);
  });

  // 작기 변경 이벤트 (전체 작기 목록에서 특정 작기 선택 시 해당 시설로 자동 동기화)
  cropSelect.addEventListener('change', (e) => {
    const selectedCropSn = e.target.value;
    cropState.filters.crop_sn = selectedCropSn;
    cropState.filters.sample_num = 'ALL';

    if (selectedCropSn !== 'ALL') {
      const matchRow = rawRows.find(r => r.crop_sn === selectedCropSn);
      if (matchRow && matchRow.facility_id) {
        cropState.filters.facility_id = matchRow.facility_id;
        facilitySelect.value = matchRow.facility_id;
      }
    }

    updateDropdownOptions();
    updateCropView(cropKey);
  });

  // 개체 변경 이벤트
  sampleSelect.addEventListener('change', (e) => {
    if (cropState.mode === 'average') {
      cropState.filters.sample_num = 'ALL';
      sampleSelect.value = 'ALL';
      return;
    }
    cropState.filters.sample_num = e.target.value;
    updateCropView(cropKey);
  });

  // 날짜 범위 변경 이벤트
  startDateInput.addEventListener('change', (e) => {
    cropState.filters.startDate = e.target.value;
    updateCropView(cropKey);
  });

  endDateInput.addEventListener('change', (e) => {
    cropState.filters.endDate = e.target.value;
    updateCropView(cropKey);
  });

  // 초기화 버튼 이벤트
  resetBtn.addEventListener('click', () => {
    cropState.filters = {
      facility_id: 'ALL',
      crop_sn: 'ALL',
      sample_num: 'ALL',
      startDate: '',
      endDate: ''
    };
    startDateInput.value = '';
    endDateInput.value = '';
    updateDropdownOptions();
    syncSampleControlState();
    updateCropView(cropKey);
    showToast('필터가 초기화되었습니다.');
  });

  // 모드 전환 이벤트 (평균 vs 개체별)
  const modeBtns = modeGroup.querySelectorAll('.mode-btn');
  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      cropState.mode = btn.dataset.mode;
      syncSampleControlState();
      updateCropView(cropKey);
    });
  });
}

// 작물별 뷰 및 4개 차트 업데이트
function updateCropView(cropKey) {
  const cropState = state[cropKey];
  const { rawRows, filters, mode, metrics } = cropState;

  // 개체당 평균 모드인 경우 실제 데이터 필터에서도 특정 개체 선택 강제 제거
  if (mode === 'average') {
    filters.sample_num = 'ALL';
  }

  // 1. 행 필터링
  const filtered = filterRows(rawRows, filters);
  cropState.filteredRows = filtered;

  // 2. 헤더 뱃지 및 상태 텍스트 갱신
  const badgeEl = document.getElementById(`${cropKey}_filter_count_badge`);
  if (badgeEl) {
    badgeEl.textContent = `조회 결과: ${filtered.length.toLocaleString()}건 / 전체 ${rawRows.length.toLocaleString()}건`;
  }

  // 선택된 작기 라벨 구하기
  let selectedCropLabel = '전체 작기';
  if (filters.crop_sn !== 'ALL') {
    const matchingRow = rawRows.find(r => r.crop_sn === filters.crop_sn);
    if (matchingRow && matchingRow.crop_start_date && matchingRow.crop_end_date) {
      selectedCropLabel = `${matchingRow.crop_start_date} ~ ${matchingRow.crop_end_date}`;
    } else {
      selectedCropLabel = `작기 ${filters.crop_sn}`;
    }
  }

  const statusEl = document.getElementById(`${cropKey}_filter_status`);
  if (statusEl) {
    const facText = filters.facility_id === 'ALL' ? '전체 시설' : filters.facility_id;
    const sampleText = (mode === 'average' || filters.sample_num === 'ALL') ? '전체 개체' : `#${filters.sample_num}`;

    if (mode === 'average') {
      statusEl.innerHTML = `<span>ℹ️ <strong>개체당 평균 모드</strong> (${facText} / ${selectedCropLabel} / ${sampleText})</span>`;
    } else {
      statusEl.innerHTML = `<span>🔍 <strong>개체별 추이 모드</strong> (${facText} / ${selectedCropLabel} / ${sampleText} 실제 관측값)</span>`;
    }
  }

  // 3. 차트 렌더링 및 개별 카드 뱃지 업데이트
  if (mode === 'average') {
    const aggregated = aggregateAverageTrend(filtered, metrics);
    metrics.forEach(mKey => {
      const cardBadge = document.getElementById(`badge_${mKey}`);
      if (cardBadge) {
        cardBadge.textContent = '개체당 평균';
        cardBadge.style.background = '#eff6ff';
        cardBadge.style.color = '#2563eb';
        cardBadge.style.borderColor = '#bfdbfe';
      }
      chartManager.renderAverageChart(mKey, aggregated[mKey]);
    });
  } else {
    const individualGroups = extractIndividualTrends(filtered, metrics);
    metrics.forEach(mKey => {
      const groupCount = individualGroups[mKey] ? individualGroups[mKey].length : 0;
      const cardBadge = document.getElementById(`badge_${mKey}`);
      if (cardBadge) {
        cardBadge.textContent = `개체별 실제값 (${groupCount}개 계열)`;
        cardBadge.style.background = '#f3e8ff';
        cardBadge.style.color = '#7e22ce';
        cardBadge.style.borderColor = '#d8b4fe';
      }
      chartManager.renderIndividualChart(mKey, individualGroups[mKey]);
    });
  }
}

// DOM 로드 완료 시 앱 실행
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});
