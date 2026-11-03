"""Export saved selection audits for display only; no fitting or result writes.

Use a new --output directory. Publication remains a separate, reviewed operation.
Only variable definitions and selection evidence are exported, never row identifiers.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REASONS = {
    'missing_over_50_percent': '결측률 50% 초과',
    'constant_or_empty': '상수 또는 유효값 없음',
    'exact_duplicate': '완전 중복',
    'pearson_abs_gte_0.95': '절대 상관계수 0.95 이상',
    'vif_over_10': 'VIF 10 초과',
}
STATS = {'mean': '평균', 'std': '표준편차(ddof=0)', 'min': '최솟값', 'max': '최댓값'}
CONTEXT = {
    'cropping_system': ('재배 시스템', ''), 'cultivation_area': ('재배면적', '평'),
    'cal_cultivation_area': ('환산 재배면적', '㎡'), 'plant_num': ('전체 재식수량', '주'),
    'cal_plant_num': ('면적당 재식수량', '주/㎡'), 'plant_density': ('재식밀도', ''),
    'cropping_season_name': ('작기명', ''), 'days_since_crop_start': ('정식 후 경과일', '일'),
}
GROWTH = {
    'leaf_len': ('엽장', 'cm'), 'leaf_cnt': ('엽수', '개'), 'leaf_wdth': ('엽폭', 'cm'),
    'ptl_len': ('엽병장', ''), 'stem_dmtr': ('줄기 직경', 'mm'), 'crwn_dmtr': ('관부 직경', ''),
    'plnt_hgt': ('초장', 'cm'), 'grw_len': ('생장길이', 'cm'), 'flclstr_hgt': ('화방 높이', 'cm'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def audit_prefix(path):
    # Correlation matrices can be hundreds of MB. Selection and exclusion records
    # precede them; read only this complete JSON prefix from the immutable file.
    before = path.stat()
    prefix = ''
    with path.open() as stream:
        while '"correlations":' not in prefix:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError(f'Missing audit boundary: {path.name}')
            prefix += chunk
            if len(prefix) > 10_000_000:
                raise ValueError('Unexpected audit layout')
    prefix = prefix.split('"correlations":', 1)[0].rstrip().rstrip(',')
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return json.loads(prefix + '}'), hashlib.sha256(prefix.encode()).hexdigest()


def definition(name, group, crop, sensor, semantic):
    target = '개화수' if crop == 'tomato' else '착과수'
    limit = 20 if crop == 'tomato' else 5
    item = dict(variable_id=name, group=group, base_variable=name, window='현재 조사 시점',
                statistic='원값', source_name='', unit='', meaning='', variable_type='수치형')
    if name == 'truss':
        item.update(group='공통', source_name='예측 대상 화방 번호', unit='화방',
                    meaning=f'예측 대상 화방의 번호(1~{limit}). E1~E7 포함 여부와 무관한 공통 입력입니다.')
    elif group == 'E1':
        item.update(source_name=f'동일 개체의 화방별 {target}', unit='개')
        descriptions = {
            'current': (f'현재 조사한 대상 화방의 {target}', '현재값'),
            'delta': (f'대상 화방의 현재 {target} − 직전 조사값', '현재값 − 직전값'),
            'roll3_mean': ('현재를 포함한 최근 3회 조사값 중 유효값의 평균(최소 1개)', '최근 3회 평균'),
            'past_max': ('최초 조사부터 현재까지의 대상 화방 관측값 최댓값', '누적 최댓값'),
            'past_observations': ('현재까지 대상 화방에서 값이 존재한 누적 조사 횟수', '유효 관측 수'),
            'days_since_first_positive': ('현재까지 최초로 양수가 관측된 날부터 현재 조사일까지의 경과일. 양수 이력이 없으면 결측', '경과일'),
            'days_since_previous': ('동일 개체·화방의 직전 조사일부터 현재 조사일까지의 경과일', '날짜 차이'),
            'lower_neighbor': (f'현재 조사에서 바로 아래 번호 화방의 {target}. 1화방에서는 결측', '인접 하위 화방 관측값'),
            'upper_neighbor': (f'현재 조사에서 바로 위 번호 화방의 {target}. {limit}화방에서는 결측', '인접 상위 화방 관측값'),
            'observed_trusses': (f'현재 조사에서 값이 존재하는 화방 수(1~{limit}화방 범위)', '유효 화방 수'),
            'positive_trusses': (f'현재 조사에서 {target}가 0보다 큰 화방 수', '양수 화방 수'),
            'positive_min': ('현재 양수가 관측된 가장 낮은 화방 번호. 양수 화방이 없으면 결측', '양수 화방 번호 최솟값'),
            'positive_max': ('현재 양수가 관측된 가장 높은 화방 번호. 양수 화방이 없으면 결측', '양수 화방 번호 최댓값'),
            'observed_total': (f'현재 조사한 1~{limit}화방의 유효 {target} 합계. 미조사 화방은 합산에서 제외', '유효 관측값 합계'),
            'weighted_position': (f'Σ(화방 번호 × 현재 {target}) / 현재 {target} 합계. 합계가 0이면 결측', '관측값 가중 화방 위치'),
            'position_previous': ('동일 개체의 직전 조사에서 계산한 관측값 가중 화방 위치', '직전 가중 위치'),
            'position_movement': ('현재 가중 화방 위치 − 직전 조사 가중 화방 위치', '가중 위치 변화'),
            'relative_position': ('대상 화방 번호 − 현재 관측값 가중 화방 위치', '상대 화방 위치'),
            'relative_max': ('대상 화방 번호 − 현재 양수가 관측된 최고 화방 번호', '최고 양수 화방과 차이'),
        }
        if name.startswith('lag'):
            n = int(name[-1]); item.update(meaning=f'동일 개체·화방의 {n}회 전 조사 {target}', window=f'{n}회 전 조사', statistic='과거 관측값')
        else:
            item['meaning'], item['statistic'] = descriptions[name]
        if name.startswith('days_'): item.update(unit='일', window='현재까지의 조사 이력')
        if name.startswith('past_'): item['window'] = '최초 조사부터 현재까지'
        if name == 'past_observations': item['unit'] = '회'
        if name == 'roll3_mean': item['window'] = '현재 포함 최근 3회 조사'
        if name in {'delta', 'position_movement'}: item['window'] = '현재·직전 조사'
        if name == 'position_previous': item['window'] = '직전 조사'
        if name in {'positive_min','positive_max','weighted_position','position_previous','position_movement','relative_position','relative_max'}: item['unit'] = '화방'
        if name in {'observed_trusses','positive_trusses'}: item['unit'] = '화방 수'
    elif group == 'E2':
        base, kind = name.split('__'); label, unit = GROWTH[base]
        item.update(base_variable=base, source_name=label, unit=unit)
        if kind == 'current': item.update(meaning=f'현재 조사한 {label}', statistic='현재값')
        elif kind == 'delta': item.update(meaning=f'{label} 현재값 − 직전 조사값', window='현재·직전 조사', statistic='변화량')
        elif kind == 'delta_per_day': item.update(meaning=f'{label} 변화량 / 직전 조사 후 경과일', window='현재·직전 조사', statistic='일평균 변화량', unit=f'{unit}/일' if unit else '원 단위/일')
        elif kind.startswith('lag'): item.update(meaning=f'동일 개체의 {kind[-1]}회 전 {label} 관측값', window=f'{kind[-1]}회 전 조사', statistic='과거 관측값')
        else:
            match = re.fullmatch(r'roll([234])_(mean|std|min|max|slope)', kind); n, stat = match.groups()
            item.update(window=f'현재 포함 최근 {n}회 조사', statistic=STATS.get(stat, '경과일에 대한 선형회귀 기울기'))
            item['meaning'] = f"{label}의 {item['statistic']}. 최근 {n}회 모두 유효값이 있어야 계산"
            if stat == 'slope': item['unit'] = f'{unit}/일' if unit else '원 단위/일'
    elif group in {'E3','E4','E5'}:
        base, days, stat = re.fullmatch(r'(.+)__([137])d_(mean|std|min|max|active_fraction)', name).groups()
        label, unit = sensor[base]
        item.update(base_variable=base, source_name=label, unit=unit,
                    window=f'직전 {days}일 [기준일−{days}일, 기준일)', statistic=STATS.get(stat, '값이 1인 기록 비율'))
        item['meaning'] = f"조사 기준일 당일을 제외한 {label} 유효 원천 기록의 {item['statistic']}"
        if stat == 'active_fraction': item['unit'] = '비율(0~1)'
    elif group == 'E6':
        row = semantic[name]; stat = row['Statistic']; days = int(row['Window_Days'])
        label = row['Device_Family_KR'] + ' ' + row['Measurement_Type_KR']
        formulas = {
            'facility_mean': ('장치별 평균의 평균', '기록된 장치 각각의 개도율 평균을 구한 뒤 장치 간 동일 가중 평균'),
            'overall_max': ('전체 최댓값', '기록된 모든 장치의 개도율 최댓값'),
            'overall_min': ('전체 최솟값', '기록된 모든 장치의 개도율 최솟값'),
            'within_device_std': ('장치별 표준편차의 평균', '각 장치 개도율의 표준편차(ddof=0)를 구한 뒤 장치 간 평균'),
            'active_fraction_mean': ('장치별 작동 기록 비율의 평균', '장치별 유효 0/1 기록 중 값이 1인 비율의 평균. 시간 가중 비율이 아님'),
            'active_fraction_max': ('장치별 작동 기록 비율의 최댓값', '장치별 유효 0/1 기록 중 값이 1인 비율의 최댓값. 시간 가중 비율이 아님'),
            'reporting_device_count': ('기록 장치 수', '해당 기간에 유효 관측이 있는 통합 장치 코드 수'),
            'connection_rate': ('관측 장치 연결률', '기간 내 기록 장치 수 / 기준일 이전에 한 번이라도 관측된 장치 수. 알려진 장치가 없으면 결측'),
        }
        statistic, meaning = formulas.get(stat, (STATS.get(stat, stat), f'모든 장치의 유효 기록을 모은 {STATS.get(stat, stat)}'))
        unit = '%' if row['Measurement_Type'] == 'opening_percent' else ''
        if stat.startswith('active_fraction') or stat == 'connection_rate': unit = '비율(0~1)'
        if stat == 'reporting_device_count': unit = '장치 수'
        item.update(base_variable=f"{row['Device_Family']} · {row['Measurement_Type']}", source_name=label, unit=unit,
                    window=f'직전 {days}일 [기준일−{days}일, 기준일)', statistic=statistic, meaning=f'{label}: {meaning}')
    elif group == 'E7':
        label, unit = CONTEXT[name]
        item.update(source_name=label, unit=unit, meaning=f'현재 조사에 연결된 시설·작기 정보: {label}')
        if name == 'days_since_crop_start': item.update(meaning='현재 조사일 − 정식일', statistic='날짜 차이')
    else: raise ValueError(name)
    assert item['meaning'] and item['source_name'], name
    return item


def export(campaign, display, output, audit_path):
    assert not output.exists(), 'Use a new output directory'
    index = json.loads((display/'manifest.json').read_text())
    manifest = json.loads((campaign/'manifest.json').read_text())
    summary = campaign/index['mode']/'summary.json'
    assert sha(summary) == index['summary_sha256']
    sensor_frame = pd.read_csv(ROOT/'strawberry_fruit_set_prediction/data/sensor_category_cache/sensor_metadata.csv').fillna('')
    sensor = {r.variable_id: (r.com_name, r.unit) for r in sensor_frame.itertuples()}
    semantic = pd.read_csv(campaign.parent/'exhaustive_e6_semantic_v2/semantic_feature_catalog.csv').fillna('').set_index('Variable_ID').to_dict('index')
    output.mkdir(parents=True)
    release = dict(schema=1, summary_sha256=index['summary_sha256'], crops={})
    audit = dict(combinations=0, model_combinations=0, source_audits={}, counts={})
    for crop, spec in manifest['crops'].items():
        fit_ids = sorted(pd.read_parquet(campaign/crop/'train.parquet', columns=['row_id']).row_id.tolist())
        models = {}
        for model, record in index['models'][crop].items():
            path = display/record['path']; assert sha(path) == record['sha256']
            models[model] = json.loads(path.read_text())['combinations']
        combos = next(iter(models.values())); assert len(combos) == 127
        group_of = {name: group for group, names in spec['groups'].items() for name in names}
        group_of['truss'] = '공통'
        definitions = {name: definition(name, group, crop, sensor, semantic) for name, group in group_of.items()}
        payload = dict(definitions=definitions, combinations={})
        for combo_id, combo in combos.items():
            key = hashlib.sha256(json.dumps([crop, combo_id, fit_ids], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:24]
            selection, prefix_hash = audit_prefix(campaign/'selection'/f'{key}.json')
            selected = selection['selected_numeric'] + selection['selected_categorical']
            excluded = selection['exclusions']
            candidates = {'truss'} | {f for g in combo['groups'] for f in spec['groups'][g]}
            assert len(selected) == len(set(selected))
            assert len(excluded) == len({r['Variable_ID'] for r in excluded})
            assert set(selected).isdisjoint(r['Variable_ID'] for r in excluded)
            assert set(selected) | {r['Variable_ID'] for r in excluded} == candidates
            for records in models.values():
                assert records[combo_id]['selected'] == selected
                assert records[combo_id]['features'] == len(selected)
                assert records[combo_id]['groups'] == combo['groups']
                audit['model_combinations'] += 1
            # Exclusion records contain only variable names and fit-only statistics.
            allowed = {'Variable_ID','Reason','Missing_Rate','Related_Variable','Correlation','Iteration','VIF','VIF_Is_Infinite'}
            for row in excluded:
                assert set(row) <= allowed and row['Reason'] in REASONS
            payload['combinations'][combo_id] = dict(groups=combo['groups'], selected=selected,
                categorical=selection['selected_categorical'], candidate_count=len(candidates), exclusions=excluded)
            audit['source_audits'][key] = prefix_hash; audit['combinations'] += 1
        path = output/f'{crop}.json'; write(path, payload)
        release['crops'][crop] = dict(path=path.name, sha256=sha(path))
        audit['counts'][crop] = dict(definitions=len(definitions), combinations=len(payload['combinations']), bytes=path.stat().st_size)
    assert audit['combinations'] == 254 and audit['model_combinations'] == 1524
    write(output/'manifest.json', release)
    audit['status'] = 'passed'; write(audit_path, audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'source_audits'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--display', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    export(args.campaign, args.display, args.output, args.audit)
