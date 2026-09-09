/**
 * CSV 파서 및 데이터 정규화 모듈
 * UTF-8 BOM 대응, 타입 캐스팅, 결측치(null)와 0 구분, 중복 키 검증
 */

export function parseCSV(csvText) {
  if (!csvText || typeof csvText !== 'string') {
    throw new Error('유효한 CSV 텍스트가 전달되지 않았습니다.');
  }

  // 1. UTF-8 BOM 제거 (\uFEFF)
  let cleanText = csvText;
  if (cleanText.charCodeAt(0) === 0xFEFF) {
    cleanText = cleanText.slice(1);
  }

  // 2. 줄바꿈 분리 (\r\n 또는 \n)
  const lines = cleanText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) {
    throw new Error('CSV에 헤더와 데이터 행이 충분하지 않습니다.');
  }

  // 3. 헤더 파싱
  const headerLine = lines[0];
  const headers = parseCSVLine(headerLine).map(h => h.trim());

  // 4. 데이터 행 파싱
  const rows = [];
  const seenKeys = new Map();
  const duplicateWarnings = [];

  for (let i = 1; i < lines.length; i++) {
    const rawLine = lines[i];
    if (!rawLine.trim()) continue;

    const values = parseCSVLine(rawLine);
    if (values.length !== headers.length) {
      console.warn(`[Line ${i + 1}] 헤더 열 수(${headers.length})와 데이터 열 수(${values.length}) 불일치:`, rawLine);
    }

    const rowObj = {};
    for (let h = 0; h < headers.length; h++) {
      const col = headers[h];
      const val = values[h] !== undefined ? values[h].trim() : '';
      rowObj[col] = val;
    }

    // 데이터 정규화
    const normalized = normalizeRow(rowObj, i + 1);
    rows.push(normalized);

    // 중복 체크: timestamp + facility_id + crop_sn + sample_num
    const dupKey = `${normalized.timestamp}__${normalized.facility_id}__${normalized.crop_sn}__${normalized.sample_num}`;
    if (seenKeys.has(dupKey)) {
      duplicateWarnings.push({
        key: dupKey,
        line: i + 1,
        firstLine: seenKeys.get(dupKey),
        timestamp: normalized.timestamp,
        facility_id: normalized.facility_id,
        crop_sn: normalized.crop_sn,
        sample_num: normalized.sample_num
      });
    } else {
      seenKeys.set(dupKey, i + 1);
    }
  }

  return {
    headers,
    rows,
    duplicateWarnings,
    totalRows: rows.length
  };
}

/**
 * 콤마 분리 및 큰따옴표 이스케이프 처리
 */
function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current);
  return result;
}

/**
 * 행 데이터 정규화:
 * - 식별자(facility_id, crop_sn, sample_num 등)는 문자열로 처리
 * - 수치 관측값은 빈칸일 때 null, 0 이상일 때 number로 처리 (0과 결측치 엄격 구분)
 * - 날짜 필드는 문자열(YYYY-MM-DD)
 */
function normalizeRow(raw, lineNum) {
  const result = {
    _line: lineNum,
    timestamp: String(raw.timestamp || '').trim(),
    split: String(raw.split || '').trim(),
    facility_id: String(raw.facility_id || '').trim(),
    crop_sn: String(raw.crop_sn || '').trim(),
    sample_num: String(raw.sample_num || '').trim(),
    cropping_serl_no: String(raw.cropping_serl_no || '').trim(),
    crop_start_date: String(raw.crop_start_date || '').trim(),
    crop_end_date: String(raw.crop_end_date || '').trim(),
    matched_crop_sn: String(raw.matched_crop_sn || '').trim()
  };

  // 고유 개체 식별자 생성: facility_id + crop_sn + sample_num
  result.entity_id = `${result.facility_id} | ${result.crop_sn} | #${result.sample_num}`;
  result.facility_crop_id = `${result.facility_id} | ${result.crop_sn}`;

  // 토마토 지표 처리
  if ('tomato_flower_truss1' in raw) {
    result.tomato_flower_truss1 = parseNumericValue(raw.tomato_flower_truss1);
  }
  if ('tomato_flower_truss2' in raw) {
    result.tomato_flower_truss2 = parseNumericValue(raw.tomato_flower_truss2);
  }
  if ('tomato_flower_truss3' in raw) {
    result.tomato_flower_truss3 = parseNumericValue(raw.tomato_flower_truss3);
  }
  if ('tomato_flower_total' in raw) {
    result.tomato_flower_total = parseNumericValue(raw.tomato_flower_total);
  }

  // 딸기 지표 처리
  if ('strawberry_fruit_truss1' in raw) {
    result.strawberry_fruit_truss1 = parseNumericValue(raw.strawberry_fruit_truss1);
  }
  if ('strawberry_fruit_truss2' in raw) {
    result.strawberry_fruit_truss2 = parseNumericValue(raw.strawberry_fruit_truss2);
  }
  if ('strawberry_fruit_truss3' in raw) {
    result.strawberry_fruit_truss3 = parseNumericValue(raw.strawberry_fruit_truss3);
  }
  if ('strawberry_fruit_total' in raw) {
    result.strawberry_fruit_total = parseNumericValue(raw.strawberry_fruit_total);
  }

  return result;
}

/**
 * 수치 파싱:
 * - 빈 문자열, null, undefined -> null (미관측/결측)
 * - 숫자로 변환 가능한 경우 -> Number (0은 0으로 반환)
 */
function parseNumericValue(val) {
  if (val === null || val === undefined) return null;
  const trimmed = String(val).trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'nan' || trimmed.toLowerCase() === 'null') {
    return null;
  }
  const num = Number(trimmed);
  return isNaN(num) ? null : num;
}
