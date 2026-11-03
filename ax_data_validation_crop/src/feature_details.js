// Display-only adapter for saved fit-time selection audits. Never recompute selection.
const REASONS = {
  missing_over_50_percent: '결측률 50% 초과', constant_or_empty: '상수 또는 유효값 없음',
  exact_duplicate: '완전 중복', 'pearson_abs_gte_0.95': '절대 상관계수 0.95 이상', vif_over_10: 'VIF 10 초과'
};
const url = path => new URL(`./feature_details/${path}`, import.meta.url);
const digest = async text => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))), b => b.toString(16).padStart(2, '0')).join('');
const crops = new Map();

async function load(crop, summaryHash) {
  const manifestResponse = await fetch(url('manifest.json'), { cache: 'no-store' });
  if (!manifestResponse.ok) throw new Error('현재 결과의 변수 설명·제외 기록이 아직 게시되지 않았습니다.');
  const manifest = await manifestResponse.json();
  if (manifest.schema !== 1 || manifest.summary_sha256 !== summaryHash) {
    throw new Error('현재 결과와 변수 상세 기록의 버전이 달라 표시를 보류합니다. 이전 실험의 제외 사유는 사용하지 않습니다.');
  }
  const spec = manifest.crops[crop];
  if (!spec) throw new Error('해당 작물의 변수 상세 기록이 없습니다.');
  const response = await fetch(url(spec.path), { cache: 'no-store' });
  if (!response.ok) throw new Error('변수 상세 기록을 불러오지 못했습니다.');
  const text = await response.text();
  if (await digest(text) !== spec.sha256) throw new Error('변수 상세 기록의 파일 검증에 실패했습니다. 새로고침해 주세요.');
  return JSON.parse(text);
}

export async function featureSelectionDetail(crop, comboId, current, summaryHash) {
  const key = `${summaryHash}/${crop}`;
  if (!crops.has(key)) crops.set(key, load(crop, summaryHash).catch(error => { crops.delete(key); throw error; }));
  const data = await crops.get(key), audit = data.combinations[comboId];
  if (!audit || JSON.stringify(audit.selected) !== JSON.stringify(current.selected) ||
      JSON.stringify(audit.groups) !== JSON.stringify(current.groups)) {
    throw new Error('현재 모델의 실제 입력 목록과 상세 기록이 일치하지 않아 표시를 보류합니다.');
  }
  const describe = name => {
    const item = data.definitions[name];
    if (!item) throw new Error(`변수 정의가 없습니다: ${name}`);
    return { ...item, variable_type: audit.categorical.includes(name) ? '범주형' : item.variable_type };
  };
  const selected = audit.selected.map(describe);
  const excluded = audit.exclusions.map(row => ({
    ...describe(row.Variable_ID), reason: row.Reason, reason_label: REASONS[row.Reason] || '기록 확인 필요',
    missing_rate: row.Missing_Rate, related_variable: row.Related_Variable,
    related_variable_name: row.Related_Variable ? data.definitions[row.Related_Variable]?.source_name : null,
    correlation: row.Correlation, iteration: row.Iteration, vif: row.VIF, vif_is_infinite: row.VIF_Is_Infinite
  }));
  const counts = new Map();
  for (const item of excluded) counts.set(item.reason, (counts.get(item.reason) || 0) + 1);
  return { groups: audit.groups, selected, excluded, candidate_count: audit.candidate_count,
    selected_count: selected.length, excluded_count: excluded.length,
    source_label: '현재 화방 확장 실험의 저장된 학습 변수 선택·제외 기록',
    exclusion_note: excluded.length ? '현재 실험의 학습 데이터에서 기록된 제외 사유만 표시합니다. 결측률·상관계수·VIF는 당시 저장된 값입니다.' : '',
    exclusion_summary: Array.from(counts, ([reason, count]) => ({ reason_label: REASONS[reason], count })) };
}
