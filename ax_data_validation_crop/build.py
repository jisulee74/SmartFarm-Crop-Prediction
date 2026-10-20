from pathlib import Path
import hashlib,json
import pandas as pd
from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent/'ax_data_validation'; DATA=ROOT/'ax_direction/screening_analysis'
selected=pd.read_csv(DATA/'selected_models.csv');metrics=pd.read_csv(DATA/'model_comparison.csv');records=[]
for row in selected.itertuples():
 d=metrics[(metrics.target==row.target)&(metrics.model==row.model)&(metrics.group_id==row.group_id)&(metrics.split=='test')&(metrics.variant=='bounded')]
 assert len(d)==1
 r=d.iloc[0];assert r.seed_count==3
 records.append({'target':row.target,'model':row.model,'group':row.categories,'n':int(r.n),'metrics':{k:float(r[k+'_mean']) for k in ['rmse','mae','r2','ccc']}})
bundle={'campaign':'exhaustive_e6_semantic_screening_v1','split':'test','seed_count':3,'aggregation':'arithmetic_mean_of_first_second_third_metrics_excluding_sum123','source_hashes':{f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['selected_models.csv','model_comparison.csv']},'targets':records}
(OUT/'summary.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
soup=BeautifulSoup((ROOT/'index.html').read_text(),'html.parser');base=soup.new_tag('base',href='../ax_data_validation/');soup.head.insert(0,base);soup.title.string='AX 데이터 구축·검증 | 작물별 성능 요약'
css=soup.new_tag('link',rel='stylesheet',href='../ax_data_validation_crop/src/summary.css');soup.head.append(css)
main=soup.select_one('#tab-experiments > main');advanced=soup.new_tag('details',id='crop_advanced',attrs={'class':'crop-details'});sm=soup.new_tag('summary');sm.string='실험 상세 탐색 · 모델·변수군·화방별 예측과 오차 분석';advanced.append(sm)
for child in list(main.contents):advanced.append(child.extract())
main.append(advanced)
labels={'tomato':'토마토 꽃수(개화수)','strawberry':'딸기 착과수'};ml={'rmse':'RMSE ↓','mae':'MAE ↓','r2':'R² ↑','ccc':'CCC ↑'};models={'tabm':'TabM','random_forest':'Random Forest','mlp':'MLP','poisson':'Poisson','catboost':'CatBoost','tft':'TFT'}
header='''<section class="crop-summary"><span class="crop-eyebrow">AI 예측모델 검증 · 작물별 요약</span><h1>작물별 예측 성능을 한눈에</h1><p>화방별로 선정한 모델의 Test 성능을 요약합니다.</p><div class="crop-switch" role="group" aria-label="요약 작물 선택"><button type="button" data-summary-crop="tomato" aria-pressed="true">토마토 꽃수</button><button type="button" data-summary-crop="strawberry" aria-pressed="false">딸기 착과수</button></div><p class="crop-status">잠정 선별 결과 · 학습과 평가 시설 분리 · Test 3시드 지표 평균</p>'''
for crop,title in labels.items():
 rows=[r for r in records if r['target'].startswith(crop+'_')]; ordered=[next(r for r in rows if r['target']==crop+'_'+part) for part in ['first','second','third','sum123']];mean={k:sum(r['metrics'][k] for r in ordered[:3])/3 for k in ml}
 header+=f'<section data-summary-panel="{crop}"'+(' hidden' if crop=='strawberry' else '')+f'><h2>{title} 예측 성능</h2><div class="crop-metrics" aria-live="polite">'
 for k,label in ml.items():header+=f'<article class="crop-metric"><span>{label}</span><strong data-metric="{k}">{mean[k]:.3f}</strong><small>1·2·3화방 지표의 산술평균</small></article>'
 header+='</div><p class="crop-explanation">각 화방의 3시드 평균 지표를 동일한 비중으로 평균했습니다. 화방마다 선정 모델·입력 조합이 다를 수 있습니다. 전체 관측을 합쳐 다시 계산한 지표나 하나의 작물 통합 모델 성능은 아닙니다. 1~3화방 합계 모델은 평균에서 제외했습니다.</p>'
 if crop=='tomato':header+='<p class="crop-caution">평균 R²가 음수입니다. 이 요약에서도 토마토 예측의 설명력은 낮으며, 개화가 끝난 하위 화방의 0 편중과 평가 구성을 함께 살펴야 합니다.</p>'
 header+='<details class="crop-details"><summary>화방별·1~3화방 합계 성능 자세히 보기</summary><div class="crop-table-scroll"><table><caption>동일한 Test·3시드 평균 기준. 합계는 상위 화방을 포함한 작물 전체 합계가 아닙니다.</caption><thead><tr><th>예측 대상</th><th>선정 모델</th><th>입력 변수군</th><th>평가 건수</th><th>RMSE</th><th>MAE</th><th>R²</th><th>CCC</th></tr></thead><tbody>'
 for part,r in zip(['1화방','2화방','3화방','1~3화방 합계'],ordered):
  header+=f'<tr><th scope="row">{part}</th><td>{models[r["model"]]}</td><td>{r["group"]}</td><td>{r["n"]:,}</td>'+''.join(f'<td>{r["metrics"][k]:.3f}</td>' for k in ml)+'</tr>'
 header+='</tbody></table></div></details></section>'
header+='</section>';main.insert(0,BeautifulSoup(header,'html.parser'))
script=soup.new_tag('script',type='module',src='../ax_data_validation_crop/src/summary.js');soup.body.append(script)
(OUT/'index.html').write_text(str(soup))
print({crop:{k:sum(r['metrics'][k] for r in records if r['target'].startswith(crop+'_') and not r['target'].endswith('sum123'))/3 for k in ml} for crop in labels})
