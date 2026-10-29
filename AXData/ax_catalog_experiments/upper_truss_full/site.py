"""Render only validated bundles, preserving the existing site's other tabs."""
import html
import json
from pathlib import Path
from bs4 import BeautifulSoup
from .prepare import ROOT

LABELS={'tomato':'토마토 개화수','strawberry':'딸기 착과수'}
MODEL={'poisson':'Poisson','random_forest':'Random Forest','catboost':'CatBoost',
       'mlp':'MLP','tabm':'TabM','tft':'TFT'}
METRICS=['rmse','mae','r2','ccc']


def number(x):
    return '산출 불가' if x is None else f'{x:.3f}'


def cells(values):
    return ''.join(f'<td>{html.escape(str(v))}</td>' for v in values)


def table(headers, rows):
    return '<div class="crop-table-scroll"><table><thead><tr>'+''.join(
        f'<th>{html.escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells(row)+'</tr>' for row in rows)+'</tbody></table></div>'


def build(bundle, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    soup=BeautifulSoup((ROOT/'ax_data_validation_crop/index.html').read_text(),'html.parser')
    main=soup.select_one('#tab-experiments > main')
    old=main.select_one('.crop-summary')
    if old:old.decompose()
    advanced=main.select_one('#crop_advanced')
    if advanced:
        advanced.find('summary').string='과거 1~3화방 실험 참고 · 현재 확장 결과와 별도 자료'
        advanced.attrs.pop('open',None)
    stage='1시드 간략 전수조합 결과' if bundle['mode']=='quick' else '3시드·3분할 시간순 교차검증 전수조합 결과'
    text=f'''<section class="crop-summary"><span class="crop-eyebrow">AI 예측모델 검증 · 화방 확장</span>
    <h1>작물별 예측 성능</h1><p class="crop-status">{stage} · E1~E7 127개 조합 × 6개 모델 × 2작물</p>
    <p>검증셋에서 선정한 작물별 공통 모델의 평가 성능입니다. 기존 데이터 평가이며 신규 AX 데이터의 효과 검증 결과가 아닙니다.</p>
    <div class="crop-switch" role="group" aria-label="요약 작물 선택">
    <button type="button" data-summary-crop="tomato" aria-pressed="true">토마토 개화수</button>
    <button type="button" data-summary-crop="strawberry" aria-pressed="false">딸기 착과수</button></div>'''
    for crop,spec in bundle['crops'].items():
        chosen=next(m for m in spec['models'] if m['model']==spec['selected'])
        text+=f'<section data-summary-panel="{crop}"'+(' hidden' if crop=='strawberry' else '')+f'><h2>{LABELS[crop]} · 1~{spec["limit"]}화방</h2>'
        text+=f'<p>선정 모델: {MODEL[chosen["model"]]} · 입력: {chosen["groups"]} · 평가 {spec["counts"]["test"]["rows"]:,}건</p><div class="crop-metrics">'
        for k,label in zip(METRICS,['RMSE ↓','MAE ↓','R² ↑','CCC ↑']):
            x=chosen['macro'][k]
            text+=f'<article class="crop-metric"><span>{label}</span><strong>{number(x["mean"])}</strong><small>화방별 지표 산술평균'
            if x['std'] is not None:text+=f' · 시드 표준편차 {x["std"]:.3f}'
            text+='</small></article>'
        text+='</div><p class="crop-explanation">각 시드에서 화방별 지표를 동일 비중으로 평균한 뒤 시드 간 평균을 계산했습니다. 전체 행을 모아 계산한 지표 및 화방 합계 예측 성능과 다릅니다. R²가 정의되지 않는 화방은 해당 평균에서 제외하며 상세 자료에 표시합니다.</p>'
        text+='<p class="crop-caution">기존에 확인한 시설 평가셋을 재사용했습니다. 신규 시설·작기 일반화가 입증된 결과는 아닙니다. 화방별 사례 수와 현재값 유지 기준선을 함께 확인하세요.</p>'
        rows=[]
        for r in chosen['per_truss']:
            b=next(x for x in spec['baselines']['persistence']['per_truss'] if x['truss']==r['truss'])
            rows.append([f'{r["truss"]}화방',r['n'],r['positive_n']]+[number(r['metrics'][k]['mean']) for k in METRICS]+[number(b['rmse'])])
        text+='<details class="crop-details"><summary>화방별·전체 합계·기준선 자세히 보기</summary>'
        text+=table(['화방','평가 건수','양수 건수','RMSE','MAE','R²','CCC','현재값 유지 RMSE'],rows)
        text+='<h3>전체 관측을 모아 계산한 성능</h3>'
        text+=table(['모델','RMSE','MAE','R²','CCC'],[
            [MODEL[chosen['model']]]+[number(chosen['pooled'][k]['mean']) for k in METRICS],
            *[[name]+[number(spec['baselines'][key]['pooled'][k]) for k in METRICS]
              for key,name in [('persistence','현재값 유지'),('zero','항상 0')]]])
        total=chosen['sum']
        text+='<h3>평가 범위 내 화방 합계</h3><p>모든 대상 화방의 정답이 같은 다음 조사일에 존재하는 사례에서 화방별 예측을 합산했습니다. 합계를 직접 학습한 별도 모델의 성능은 아닙니다.</p>'
        text+=table(['완전 관측 건수','RMSE','MAE','R²','CCC'],[[total['n']]+[number(total['metrics'][k]['mean']) for k in METRICS]]) if total else '<p>전체 화방의 정답이 갖춰진 조사쌍이 없어 합계 성능을 산출하지 않았습니다.</p>'
        text+='</details><details class="crop-details"><summary>6개 모델별 선정 조합의 평가 성능</summary>'
        text+=table(['모델','입력 조합','검증 RMSE','평가 RMSE','평가 MAE','평가 R²','평가 CCC'],
            [[MODEL[m['model']],m['groups'],number(m['validation_rmse'])]+[number(m['pooled'][k]['mean']) for k in METRICS] for m in spec['models']])
        text+='</details><details class="crop-details"><summary>127개 입력 조합 전체 비교 · 검증셋 지표</summary><p>조합 선정에 사용한 검증셋 결과입니다. 평가셋 결과와 구분하세요. 화방 번호는 모든 조합의 공통 입력이며, E1의 생육 이력은 E1 포함 조합에만 들어갑니다.</p>'
        text+=table(['모델','입력 조합','선택 변수 수','RMSE','MAE','R²','CCC'],[
            [MODEL[r['model']],r['groups'],r['features']]+[number(r['metrics'][k]['mean']) for k in METRICS]
            for r in bundle['comparison'] if r['crop']==crop])
        text+='</details></section>'
    text+='<p><a href="../ax_data_validation_crop/summary.json">결과·선정 근거 다운로드</a> · <a href="../ax_data_validation_crop/validation_comparison.csv">전체 조합 검증 지표 CSV</a></p></section>'
    main.insert(0,BeautifulSoup(text,'html.parser'))
    soup.title.string='AX 데이터 구축·검증 | 화방 확장 성능'
    (output/'index.html').write_text(str(soup))
    (output/'summary.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
    assert len(soup.select('[data-summary-panel]'))==2
    assert len(soup.select('.crop-summary .crop-metric'))==8
    assert len(soup.select('[data-summary-crop]'))==2
