from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from AXData.ax_catalog_experiments.upper_truss_screening.prepare import OUT
from AXData.ax_catalog_experiments.stability.policy import score
from AXData.ax_catalog_experiments.upper_truss_screening.run import metrics as detailed_metrics
m=json.loads((OUT/'manifest.json').read_text());f=json.loads((OUT/'frozen_winners.json').read_text());baselines=json.loads((OUT/'baselines.json').read_text());metrics=pd.read_csv(OUT/'metrics.csv');per=pd.read_csv(OUT/'per_truss_metrics.csv');runs=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*.json')]
def table(d):return d.to_markdown(index=False,floatfmt='.4f')
text=['# 상위 화방 확장 1시드 간략실험 결과','', '시드 42. 토마토 1~20화방 개화수·딸기 1~7화방 착과수 공통 모델. 4개 입력 구성 × 6개 모델 × 2작물의 고정 설정 비교. 신규 독립 검증이나 3시드 정식 실험이 아니다. 대시보드는 변경하지 않았다.','', '## 실행 상태','',json.dumps(json.loads((OUT/'progress.json').read_text()),ensure_ascii=False),'','## 선정 및 평가 원칙','', '모델별 입력 조합 및 전체 대표 모델은 Validation RMSE로 선정 후 Test 이전 동결. Test는 Train+Validation 재학습, 선택 epoch 유지. 아래 대표 모델은 Test 성능으로 다시 고른 것이 아니다. 같은 작물의 평가 행은 모든 모델에서 동일하다. 표의 RMSE·MAE·R²·CCC는 개체×화방 관측쌍을 모아 계산한 pooled 값이며, 화방별 지표 산술평균도 별도로 표시한다. 기존 1~3화방 결과와는 타깃 범위·원천·행 구성이 달라 수치만 직접 비교하지 않는다.']
comparisons=[];facility_comparisons=[]
text+=['', '## 입력 구성', '',
       '| 구성 | 후보 입력 |', '|---|---|',
       '| A_history | 해당 화방의 현재값·과거 이력·화방 번호·작기 경과일 |',
       '| B_position | A + 인접 화방 값·양수 관측 위치·수량 가중 위치·이전 조사 대비 이동 |',
       '| C_vegetative | B + 현재 영양생장 조사값 |',
       '| D_full | C + 7일 환경·근권·제어 집계 후보와 수치 작기 정보 |',
       '', '학습자료에서 결측률 50% 초과 또는 상수인 입력은 제외했다. 실제 포함된 입력은 manifest.json에 기록했다. 이번 D 구성에서 남은 제어 관련 입력은 천창 연결률이며, 제어 개도·작동량 전체의 효과를 검증한 것은 아니다. 토마토 목표는 조사 시점의 화방개화수이며 누적 꽃수와 구분한다.']
for crop in m['crops']:
 text+=['',f'## {crop}','', '### 고정된 데이터 구성','',table(pd.DataFrame(m['crops'][crop]['counts']).T.reset_index().rename(columns={'index':'split'}))]
 model=f['overall_model'][crop];w=next(w for w in f['per_model'] if w['crop']==crop and w['model']==model);test=next((r for r in runs if r['crop']==crop and r['stage']=='test' and r['model']==model),None)
 text+=['',f'Validation 선정 대표: **{model} / {w["config"]}**','', '### 모델별 선정 조합의 Test 성능','',table(metrics[(metrics.crop==crop)&(metrics.stage=='test')][['model','config','features','best_epoch','rmse','mae','r2','ccc','positive_mae','change_mae']]),'', '### 동일 평가 행의 단순 기준선','',table(pd.DataFrame([{k:v for k,v in b.items() if k not in ['per_truss','crop','stage']} for b in baselines if b['crop']==crop and b['stage']=='test']))]
 text+=['','### 4개 입력 구성 비교 (Validation RMSE)','',metrics[(metrics.crop==crop)&metrics.stage.eq('validation')].pivot(index='model',columns='config',values='rmse').to_markdown(floatfmt='.4f')]
 macro=per[per.crop.eq(crop)&per.stage.eq('test')].groupby(['model','config'])[['rmse','mae','r2','ccc']].mean().reset_index()
 text+=['', '### 화방별 네 지표의 산술평균 (Test)', '', table(macro),
         '', '각 화방을 같은 비중으로 평균했다. 표본 수에 따라 가중하지 않으며, RMSE 평균은 전체 행을 모아 계산한 RMSE와 다르다. 정의되지 않는 R²가 있으면 해당 값은 평균에서 제외된다.']
 if test and test['status']=='success':
  d=per[(per.crop==crop)&per.model.eq(model)&per.stage.eq('test')].copy()
  baseline=next(b for b in baselines if b['crop']==crop and b['stage']=='test' and b['model']=='persistence')
  baseline_by_truss={b['truss']:b for b in baseline['per_truss']}
  d['persistence_rmse']=d.truss.map(lambda t:baseline_by_truss[t]['rmse'])
  d['rmse_reduction_pct']=100*(1-d.rmse/d.persistence_rmse)
  text+=['','### 대표 모델의 화방별 Test 성능','',table(d[['truss','n','positive_n','rmse','mae','r2','ccc','positive_mae','persistence_rmse','rmse_reduction_pct']]),'', 'rmse_reduction_pct는 현재값 유지 대비 RMSE 감소율이다. 음수면 현재값 유지보다 나쁘다.', '', '화방별 지표 산술평균: '+json.dumps({k:float(d[k].mean()) for k in ['rmse','mae','r2','ccc']},ensure_ascii=False)]
  p=pd.read_parquet(OUT/'predictions'/f'{crop}__{w["config"]}__{model}__test.parquet')
  for scope,x in [('1~3',p[p.truss<=3]),('상위화방',p[p.truss>3]),('전체',p)]:
   for name,pred in [(model,x.prediction.to_numpy()),('persistence',x.current.to_numpy()),('zero',np.zeros(len(x)))]:comparisons.append({'crop':crop,'scope':scope,'model':name,**detailed_metrics(x,pred)})
  for facility,x in p.groupby('facility_id'):
   for name,pred in [(model,x.prediction.to_numpy()),('persistence',x.current.to_numpy())]:facility_comparisons.append({'crop':crop,'facility_id':facility,'model':name,**detailed_metrics(x,pred)})
text+=['','## 대표 모델과 기준선의 1~3화방 / 상위 화방 분리 비교','',table(pd.DataFrame(comparisons)), '', '## 해석 한계','', '- 한 시드·제한된 epoch·고정 설정이므로 모델 종류의 일반적 우열을 확정하지 않는다. 특히 TFT 12epoch 상한은 정식 튜닝과 다르다.\n- Test 시설은 이전 감사에서 확인한 기존 holdout이다. 최종 효과는 새 작기/시설로 검증해야 한다.\n- 양수 사례가 적은 화방에서는 높은 전체 점수만 보지 말고 양수 MAE·기준선·시설 분포를 함께 본다.\n- 상위 화방 확장과 공통 모델 변경이 함께 이루어졌다. 이전 모델과의 차이를 화방 확장만의 인과적 효과로 해석하지 않는다.\n- D 구성은 기존 데이터의 같은 날짜 환경·제어 집계를 재사용해 일부 날짜는 결측이다. 정식 실험에서는 원천 전체 재집계가 필요하다.\n- 딸기 위치는 착과 위치이며 토마토 개화 위치와 같은 생리 법칙을 강제한 것이 아니다.']
failed=[r for r in runs if r['status']!='success']
if failed:text+=['','## 실패 기록','',json.dumps(failed,ensure_ascii=False,indent=2)]
text+=['','## 평가 시설별 대표 모델과 현재값 유지 비교','',table(pd.DataFrame(facility_comparisons))]
(OUT/'결과보고서.md').write_text('\n'.join(text));pd.DataFrame(comparisons).to_csv(OUT/'scope_baseline_comparison.csv',index=False);pd.DataFrame(facility_comparisons).to_csv(OUT/'facility_baseline_comparison.csv',index=False);print(OUT/'결과보고서.md')
