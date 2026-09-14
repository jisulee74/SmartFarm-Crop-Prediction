/**
 * 데이터 분석 및 집계 엔진 모듈
 * 상단 메타데이터 계산, 개체당 평균 집계, 개체별 추이 추출, 연계 필터링
 */

/**
 * 작물 데이터셋의 전체 요약 통계 산출
 */
export function computeSummaryStats(rows) {
  if (!rows || rows.length === 0) {
    return {
      periodStart: '-',
      periodEnd: '-',
      facilityCount: 0,
      cropCycleCount: 0,
      uniqueEntityCount: 0,
      totalObservations: 0,
      cropCyclePerFacility: { min: 0, max: 0, avg: '0.0' }
    };
  }

  let minDate = rows[0].timestamp;
  let maxDate = rows[0].timestamp;

  const facilities = new Set();
  const facilityToCropSns = new Map();
  const uniqueEntities = new Set();

  for (const row of rows) {
    if (row.timestamp) {
      if (row.timestamp < minDate) minDate = row.timestamp;
      if (row.timestamp > maxDate) maxDate = row.timestamp;
    }
    if (row.facility_id) {
      facilities.add(row.facility_id);
      if (!facilityToCropSns.has(row.facility_id)) {
        facilityToCropSns.set(row.facility_id, new Set());
      }
      if (row.crop_sn) {
        facilityToCropSns.get(row.facility_id).add(row.crop_sn);
      }
    }
    if (row.entity_id) {
      uniqueEntities.add(row.entity_id);
    }
  }

  // 시설별 작기 수 계산
  const cropCounts = Array.from(facilityToCropSns.values()).map(s => s.size);
  const minCrops = cropCounts.length > 0 ? Math.min(...cropCounts) : 0;
  const maxCrops = cropCounts.length > 0 ? Math.max(...cropCounts) : 0;
  const avgCrops = cropCounts.length > 0 ? (cropCounts.reduce((a, b) => a + b, 0) / cropCounts.length).toFixed(1) : '0.0';

  let totalCropCycles = 0;
  for (const crops of facilityToCropSns.values()) {
    totalCropCycles += crops.size;
  }

  return {
    periodStart: minDate,
    periodEnd: maxDate,
    facilityCount: facilities.size,
    cropCycleCount: totalCropCycles,
    uniqueEntityCount: uniqueEntities.size,
    totalObservations: rows.length,
    cropCyclePerFacility: {
      min: minCrops,
      max: maxCrops,
      avg: avgCrops
    }
  };
}

/**
 * 드롭다운 필터 옵션 목록 추출 (연계 필터링 지원 및 작기 시작~종료일 표시)
 */
export function getFilterOptions(rows, currentFilter = {}) {
  const facilitySet = new Set();
  const allCropMap = new Map(); // 전체 작기 목록
  const filteredCropMap = new Map(); // 현재 시설 조건에 맞는 작기 목록
  const sampleNumSet = new Set();

  for (const row of rows) {
    if (row.facility_id) {
      facilitySet.add(row.facility_id);
    }

    if (row.crop_sn && !allCropMap.has(row.crop_sn)) {
      allCropMap.set(row.crop_sn, {
        crop_sn: row.crop_sn,
        facility_id: row.facility_id,
        crop_start_date: row.crop_start_date || '',
        crop_end_date: row.crop_end_date || ''
      });
    }

    const facilityMatch = !currentFilter.facility_id || currentFilter.facility_id === 'ALL' || row.facility_id === currentFilter.facility_id;
    if (facilityMatch && row.crop_sn) {
      if (!filteredCropMap.has(row.crop_sn)) {
        filteredCropMap.set(row.crop_sn, {
          crop_sn: row.crop_sn,
          facility_id: row.facility_id,
          crop_start_date: row.crop_start_date || '',
          crop_end_date: row.crop_end_date || ''
        });
      }
    }

    const cropMatch = !currentFilter.crop_sn || currentFilter.crop_sn === 'ALL' || row.crop_sn === currentFilter.crop_sn;
    if (facilityMatch && cropMatch && row.sample_num) {
      sampleNumSet.add(row.sample_num);
    }
  }

  const facilities = Array.from(facilitySet).sort();

  const formatCropList = (map) => {
    return Array.from(map.values())
      .sort((a, b) => (a.crop_start_date || '').localeCompare(b.crop_start_date || '') || a.crop_sn.localeCompare(b.crop_sn))
      .map(item => {
        let label = '';
        if (item.crop_start_date && item.crop_end_date) {
          label = `${item.crop_start_date} ~ ${item.crop_end_date}`;
        } else if (item.crop_start_date) {
          label = `${item.crop_start_date} 시작`;
        } else {
          label = `작기 ${item.crop_sn}`;
        }

        return {
          value: item.crop_sn,
          label: label,
          startDate: item.crop_start_date,
          endDate: item.crop_end_date,
          facility_id: item.facility_id
        };
      });
  };

  const cropSns = formatCropList(filteredCropMap);
  const allCropSns = formatCropList(allCropMap);

  const sampleNums = Array.from(sampleNumSet).sort((a, b) => {
    const numA = Number(a);
    const numB = Number(b);
    if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
    return String(a).localeCompare(String(b));
  });

  return { facilities, cropSns, allCropSns, sampleNums };
}

/**
 * 필터 조건에 따라 행 필터링 (split은 필터링하지 않고 모두 포함)
 */
export function filterRows(rows, filters = {}) {
  const {
    facility_id = 'ALL',
    crop_sn = 'ALL',
    sample_num = 'ALL',
    startDate = '',
    endDate = ''
  } = filters;

  return rows.filter(row => {
    if (facility_id !== 'ALL' && row.facility_id !== facility_id) return false;
    if (crop_sn !== 'ALL' && row.crop_sn !== crop_sn) return false;
    if (sample_num !== 'ALL' && row.sample_num !== sample_num) return false;
    if (startDate && row.timestamp < startDate) return false;
    if (endDate && row.timestamp > endDate) return false;
    return true;
  });
}

/**
 * 개체당 평균 시계열 데이터 집계
 * - 날짜 오름차순 정렬
 * - 결측치(null)는 유효 표본에서 제외
 * - 0은 정상 관측값으로 포함
 * - 반환 형식: { [metricKey]: [ [timestamp, avgValue, validCount, min, max, totalEntitiesThatDay], ... ] }
 */
export function aggregateAverageTrend(filteredRows, metricKeys) {
  // 1. 날짜별 그룹핑
  const dateMap = new Map();

  for (const row of filteredRows) {
    const date = row.timestamp;
    if (!date) continue;

    if (!dateMap.has(date)) {
      dateMap.set(date, []);
    }
    dateMap.get(date).push(row);
  }

  // 2. 날짜 오름차순 정렬
  const sortedDates = Array.from(dateMap.keys()).sort();

  const result = {};
  for (const key of metricKeys) {
    result[key] = [];
  }

  for (const date of sortedDates) {
    const dayRows = dateMap.get(date);
    const totalEntitiesToday = new Set(dayRows.map(r => r.entity_id)).size;

    for (const key of metricKeys) {
      let sum = 0;
      let validCount = 0;
      let min = Infinity;
      let max = -Infinity;

      for (const row of dayRows) {
        const val = row[key];
        if (val !== null && val !== undefined && typeof val === 'number') {
          sum += val;
          validCount++;
          if (val < min) min = val;
          if (val > max) max = val;
        }
      }

      if (validCount > 0) {
        const avg = sum / validCount;
        result[key].push({
          date,
          value: Number(avg.toFixed(2)),
          rawValue: avg,
          validCount,
          totalEntitiesToday,
          min,
          max
        });
      }
    }
  }

  return result;
}

/**
 * 개체별 추이 데이터 추출 (개체별 모드)
 * - 개체 단위(facility_id + crop_sn + sample_num)로 분할
 * - 시설/작기가 다른 관측점은 서로 다른 시리즈로 분리하여 선이 연결되지 않도록 보장
 * - 실제 관측값 그대로 출력
 */
export function extractIndividualTrends(filteredRows, metricKeys) {
  // 개체별(facility_id + crop_sn + sample_num)로 그룹화
  const entityGroups = new Map();

  for (const row of filteredRows) {
    const key = `${row.facility_id} | 작기: ${row.crop_sn} | 개체 #${row.sample_num}`;
    const displayName = `개체 #${row.sample_num} (${row.facility_id})`;
    if (!entityGroups.has(key)) {
      entityGroups.set(key, {
        name: displayName,
        key: key,
        facility_id: row.facility_id,
        crop_sn: row.crop_sn,
        sample_num: row.sample_num,
        crop_start_date: row.crop_start_date,
        crop_end_date: row.crop_end_date,
        rows: []
      });
    }
    entityGroups.get(key).rows.push(row);
  }

  const result = {};
  for (const mKey of metricKeys) {
    result[mKey] = [];
  }

  for (const [entityName, group] of entityGroups.entries()) {
    // 날짜 오름차순 정렬
    group.rows.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

    for (const mKey of metricKeys) {
      const points = [];
      for (const row of group.rows) {
        const val = row[mKey];
        if (val !== null && val !== undefined && typeof val === 'number') {
          points.push({
            date: row.timestamp,
            value: val,
            facility_id: row.facility_id,
            crop_sn: row.crop_sn,
            sample_num: row.sample_num,
            crop_start_date: row.crop_start_date,
            crop_end_date: row.crop_end_date
          });
        }
      }

      if (points.length > 0) {
        result[mKey].push({
          entityName,
          facility_id: group.facility_id,
          crop_sn: group.crop_sn,
          sample_num: group.sample_num,
          crop_start_date: group.crop_start_date,
          crop_end_date: group.crop_end_date,
          points
        });
      }
    }
  }

  return result;
}
