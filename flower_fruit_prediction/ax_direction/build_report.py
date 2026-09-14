"""Build the three deliverables and an offline HTML report from audited analysis tables."""
from pathlib import Path
import sys,json,datetime,zipfile
import numpy as np
import pandas as pd
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
T=HERE/'tables';F=HERE/'figures';F.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parents[3]))
from AXData.ax_catalog_experiments.report import write_xlsx
from analyze import metrics
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_manager.fontManager.addfont(FONT);plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':10})
ORDER=[f'{c}_{p}' for c in ['tomato','strawberry'] for p in ['first','second','third','sum123']]
LABEL={t:('토마토 꽃수' if t.startswith('tomato') else '딸기 착과수')+' '+{'first':'1화방','second':'2화방','third':'3화방','sum123':'1~3화방 합계'}[t.split('_')[1]] for t in ORDER}
b=pd.read_csv(T/'baseline_performance.csv');s=pd.read_csv(T/'error_strata.csv');c=pd.read_csv(T/'cohort_profile.csv')
sel=pd.read_csv(T/'selected_models.csv');pairs=pd.read_csv(T/'paired_category_comparisons.csv')
am=pd.read_csv(T/'model_seed_metrics.csv');audit=json.loads((HERE/'audit.json').read_text())
bt=b[b.split=='test'].set_index('target').loc[ORDER];bv=b[b.split=='validation'].set_index('target').loc[ORDER]
ct=c[c.split=='test'].set_index('target').loc[ORDER]
ROOT=Path(audit['source_root'])

def table(d):return d.to_markdown(index=False,floatfmt='.3f')
def write(name,text): (HERE/name).write_text(text.strip()+'\n')
def csv(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(T/(name+'.csv'),index=False,encoding='utf-8-sig');return d

def st(target,dimension,level,split='test'):
 return s[(s.target==target)&(s.split==split)&(s.dimension==dimension)&(s.level==level)].iloc[0]

# Supplemental crossed strata: descriptive control of overlapping factors, no causal identification.
pp=pd.read_parquet(T/'selected_predictions.parquet'); crossed=[]
for target in ORDER:
 d=pd.read_parquet(ROOT/'prepared'/target/'test.parquet').set_index('row_id').sort_index()
 p=pp[(pp.target_id==target)&(pp.Eval_Split=='test')].pivot(index='row_id',columns='seed',values='prediction_bounded').loc[d.index]
 horizon=(d.target_date-d.feature_date).dt.days
 keys=pd.DataFrame({'facility':d.facility_id,'age':np.where(d.days_since_crop_start<=60,'<=60','61+'),
                    'horizon':np.where(horizon<=7,'1-7','8+'),'change':np.where(d.target==d['flower_count__current' if target.startswith('tomato') else 'fruit_count__current'],'unchanged','changed')},index=d.index)
 for cols in [['facility','age'],['facility','horizon'],['facility','age','horizon'],['age','change']]:
  for values,idx in keys.groupby(cols).groups.items():
   if not isinstance(values,tuple):values=(values,)
   arr=p.loc[idx].to_numpy().T; y=d.loc[idx,'target'].to_numpy()
   crossed.append(dict(target=target,dimensions='+'.join(cols),level=' / '.join(map(str,values)),n=len(idx),
                        rmse_mean=np.mean([metrics(y,a)['rmse'] for a in arr]),small_n=len(idx)<30))
csv('crossed_error_strata',crossed)
pc=csv('category_effect_summary',pairs.groupby(['target','added_category']).agg(comparisons=('delta_rmse','size'),
 median_rmse_delta=('delta_rmse','median'),min_rmse_delta=('delta_rmse','min'),max_rmse_delta=('delta_rmse','max'),
 improved_comparisons=('delta_rmse',lambda x:int((x<0).sum()))).reset_index())
model_summary=csv('model_comparison',am.groupby(['target','split','model','group_id','variant']).agg(n=('n','first'),
 seed_count=('seed','nunique'),rmse_mean=('rmse','mean'),rmse_seed_std=('rmse','std'),mae_mean=('mae','mean'),r2_mean=('r2','mean'),ccc_mean=('ccc','mean')).reset_index())

fig,axs=plt.subplots(1,2,figsize=(13,5))
for ax,targets in zip(axs,[ORDER[:4],ORDER[4:]]):
 x=np.arange(4);d=bt.loc[targets];w=.23
 ax.bar(x-w,d.rmse_mean,w,label='검증으로 선정한 모델',color='#136f63',yerr=d.rmse_seed_std,capsize=3)
 ax.bar(x,d.persistence_rmse,w,label='현재값 유지',color='#e2a03f')
 ax.bar(x+w,d.zero_predictor_rmse,w,label='항상 0 (진단 기준)',color='#8f9aa8')
 ax.set_xticks(x,['1화방','2화방','3화방','합계']);ax.set_ylabel('테스트 RMSE (개)');ax.set_title('토마토 꽃수' if targets[0].startswith('tomato') else '딸기 착과수')
 ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axs[0].legend(fontsize=8);fig.suptitle('작은 RMSE만으로 성능을 판단하지 않고 단순 기준모델과 비교',fontsize=13)
fig.text(.5,.01,'오차막대: 5개 seed의 표준편차. 시설 일반화 신뢰구간이 아님. 두 패널의 세로축 범위가 다름.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(F/'baseline_comparison.png',dpi=180);fig.savefig(F/'baseline_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(13,5))
for ax,targets in zip(axs,[ORDER[:4],ORDER[4:]]):
 x=np.arange(4);ds=[st(t,'changed_binary','changed') for t in targets]
 ax.bar(x-.18,[a.row_share*100 for a in ds],.36,label='수량 변화 관측 비율',color='#8294a5')
 ax.bar(x+.18,[a.squared_error_share*100 for a in ds],.36,label='해당 관측의 제곱오차 비중',color='#be573f')
 ax.set_xticks(x,['1화방','2화방','3화방','합계']);ax.set_ylim(0,105);ax.set_ylabel('%');ax.set_title('토마토' if targets[0].startswith('tomato') else '딸기');ax.grid(axis='y',alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('관측 수량이 변한 구간에 오차가 집중됨 (사후 진단)',fontsize=13)
fig.tight_layout();fig.savefig(F/'change_error_concentration.png',dpi=180);fig.savefig(F/'change_error_concentration.pdf');plt.close(fig)
fig,axs=plt.subplots(2,4,figsize=(15,7))
for ax,target in zip(axs.flat,ORDER):
 q=s[(s.target==target)&(s.split=='test')&(s.dimension=='crop_age_days')].set_index('level').reindex(['<=60','61-120','121-180','181+'])
 ax.bar(np.arange(4),q.rmse_mean,color='#136f63');ax.set_xticks(np.arange(4),['≤60','61–120','121–180','181+']);ax.set_title(LABEL[target],fontsize=10);ax.set_ylabel('RMSE (개)')
 for i,(_,row) in enumerate(q.iterrows()):
  if pd.notna(row['n']):ax.text(i,row.rmse_mean,f'n={int(row["n"])}',ha='center',va='bottom',fontsize=8)
 ax.margins(y=.25)
fig.suptitle('작기 경과일별 오차 — 생리 단계의 확정 분류는 아님',fontsize=13);fig.supxlabel('예측 시점의 작기 경과일');fig.tight_layout()
fig.savefig(F/'crop_age_error.png',dpi=180);fig.savefig(F/'crop_age_error.pdf');plt.close(fig)

base_rows=[];segment_rows=[];diagnosis=[]
for t in ORDER:
 x=bt.loc[t];v=bv.loc[t];ch=st(t,'changed_binary','changed');early=st(t,'crop_age_days','<=60')
 base_rows.append({'타깃':LABEL[t],'모델':x.model,'검증 RMSE':v.rmse_mean,'테스트 RMSE':x.rmse_mean,'seed SD':x.rmse_seed_std,
                   'MAE':x.mae_mean,'R²':x.r2_mean,'CCC':x.ccc_mean,'유지 기준 RMSE':x.persistence_rmse,'유지 대비 개선 %':x.skill_vs_persistence_pct})
 segment_rows.append({'타깃':LABEL[t],'N':int(x.n),'0 비율 %':ct.loc[t].zero_rate*100,'변화 N':int(ch.n),
                      '변화 RMSE':ch.rmse_mean,'변화 구간 SSE %':ch.squared_error_share*100,'양수 N':int(ct.loc[t].positive_n)})
 if t.startswith('tomato'):
  fac=st(t,'facility','PF_0026456_01')
  finding=f'작기 ≤60일에 SSE {early.squared_error_share*100:.1f}%; PF_0026456_01 시설 {int(fac.n)}행에서 SSE {fac.squared_error_share*100:.1f}%; 0 예측보다 RMSE 큼'
  priority='화방 생리 상태·0 의미 및 초기 관측 정합성 확인; 초기 이력 확보; 시설별 추가 평가'
 else:
  inc=st(t,'change','increase_ge2')
  finding=f'2개 이상 증가 {int(inc.n)}행의 bias {inc.bias_mean:.2f}개, SSE {inc.squared_error_share*100:.1f}%; 변화 구간 오차 집중'
  priority='기존 생육 변수 활용 검증 → 수정·적과·수확 등 작업 및 화방 상태 연계'
 if t.endswith('sum123'):priority+='; 합계 산출 방식 및 단순 유지 기준 우선 비교'
 diagnosis.append({'target':t,'finding':finding,'next_validation':priority,'evidence':'baseline_performance.csv; error_strata.csv'})
csv('target_diagnosis',diagnosis)

sources='''## 근거 자료와 해석 범위

- [S1] [0826 회의록](../../../회의록_AX데이터구축_0826.pdf), 6~7항: 꽃수·착과수 모델의 검증 역할, 수확량과의 구분, 지표 정확성 확인. 이 보고서의 상대 링크 대신 `sources/회의록_AX데이터구축_0826.txt`에서 본문 확인 가능.
- [S2] [0909 회의록](../../../회의록_AX데이터구축_0909.pdf), 2~4항 및 향후 조치: 원인 구분, 지표별 설명, 개선 과업과 As-Is/To-Be 대응 요구. **파일명은 0909이지만 본문 회의일은 2026-09-10**이다. `sources/회의록_AX데이터구축_0909.txt` 제공.
- [S3] `../../outputs/stability_v2/`: 확정된 실험 데이터·선정 결과·예측 파일. 상대 경로는 이 문서 폴더 기준이다.
- [S4] `../../stability/engine.py`, `policy.py`, `data.py`: 선정 규칙, train 기반 보정, 다음 관측 타깃과 시설 분할 정의. 읽은 파일의 SHA-256은 `source_hashes.json`에 기록했다.
- 회의록은 과업 요구의 근거이며, 회의 당시의 수치·해석을 이번 8개 타깃 실험의 사실로 전용하지 않았다. 외부 문헌 검토나 신규 모델 학습은 이번 산출물 범위에 포함하지 않았다.
'''

report1=f'''# 1. 생육 예측 기준 성능 및 오차 원인 분석

분석 기준일: 2026-09-14. 완료 시점: {audit['campaign_finished_at']}. 대상: `stability_v2` 8개 예측 타깃.

## 판단 요약

1. **8개 타깃 39,856회가 모두 완료됐다.** 예측 모델의 활용 가능성은 타깃·시설·생육 구간에 따라 다르다. 기존 데이터 전체를 불량 또는 AI에 부적합하다고 결론 낼 근거는 없다.
2. **토마토는 0 중심의 표본 구성과 초기 구간·시설 편차가 핵심 진단 사항이다.** 테스트의 96.3~97.0%가 0이다. RMSE가 작아도 4개 타깃 모두 R²가 음수이고, 항상 0을 예측하는 단순 기준보다 RMSE가 크다. 단, 항상 0을 예측하는 방식도 실제 꽃 발생을 놓치므로 운영 대안으로 채택하자는 뜻은 아니다.
3. **딸기는 관측 이력의 예측 신호가 확인되지만 수량 변화 대응이 부족하다.** 개별 화방은 현재값 유지 대비 테스트 RMSE가 6.9~13.7% 감소했고 R²는 0.58~0.69다. 변화 구간에 제곱오차의 91.6~96.0%가 집중된다.
4. **1~3화방 합계는 별도 개선이 필요하다.** 검증으로 선정한 직접 합계 모델은 유지 기준보다 토마토 RMSE가 21.0%, 딸기가 0.54% 크다. 테스트에서 더 좋아 보이는 다른 모델로 사후 교체하지 않았다.
5. **추가 수집 효과는 아직 검증되지 않았다.** 기존 미활용 변수 확인 → 관측·연결 정합성 점검 → 작업·상태 데이터 추가 → 새 평가 구간의 비교 검증 순서가 적절하다.

## 결과 신뢰성 점검

- 실행 인덱스 39,856개와 실제 result 파일 ID 집합 일치, 상태 전부 success. 24개 prepared 데이터 파일의 해시와 확정 선정 파일의 해시 일치.
- 모델별 최적 변수군 6종 × 5 seeds × 8 targets × 2 splits 및 유지 기준 16개 = **496개 예측 파일**의 row_id·시설·작기·개체·날짜·정답을 원본 평가 표본과 대조했다.
- raw/bounded의 RMSE·MAE·R²·CCC **3,968개 지표를 재계산해 일치**를 확인했다. 총 {audit['checks']:,}개 검사 통과. `audit.json`, `tables/integrity_checks.csv` 참조.
- 그 외 후보의 변수군 비교는 해시를 기록한 `Results_Summary.csv`를 사용했다. 39,856개 예측 파일 전부를 재계산한 것은 아니다.
- 공용 `progress.json`에 남은 예전 heartbeat의 running 표시는 종료 판정에 쓰지 않았다. scheduler와 8개 status 및 완료 결과가 기준이다.

## 평가 정의

타깃은 같은 시설·작기·개체의 **다음 실제 조사 시점**의 1·2·3화방 수량 또는 그 합계이다. 고정 7일 후 예측이 아니며, 전체 화방 수량·수확량 예측과도 다르다. 합계에는 1~3화방 관측이 모두 있는 표본만 사용한다. 타깃별 N이 다르므로 서로 다른 타깃의 RMSE를 직접 순위화하지 않는다.

내부 시간 교차검증으로 하이퍼파라미터를 정하고, 시설이 분리된 validation의 bounded RMSE 5개 seed 평균으로 모델·변수군을 선정했다. 동률 순서는 MAE, 변수 수, ID 등 실제 코드의 규칙을 따른다. 이 보고서는 `selection_frozen.json`의 overall을 그대로 사용한다. 각 모델별 최적 결과 전체는 `tables/model_comparison.csv`에 있다.

주 지표는 **각 seed의 지표를 구한 뒤 평균**한 값이며, 표준편차는 학습 초기화 변동이다. 예측 평균으로 앙상블을 만든 뒤 산출한 지표가 아니다. 구간별 SSE 비중은 각 관측의 제곱오차를 seed 평균한 뒤 합산해 계산한다. 유지 기준은 예측 시점의 현재 수량을 그대로 다음 값으로 사용하는 persistence이다. 0 기준은 같은 표본에서 새로 계산한 진단용 상수 모델이다.

**현재 테스트는 이미 확인한 holdout**이므로 이번 분석은 탐색·진단용이다. 시설은 분리됐지만 전체 기간의 전향적 시간 분할은 아니다. 학습 자료에 테스트 시점 이후의 다른 시설 관측도 있으므로 미래 운영 성능을 입증하지 않는다. 관련 판단은 새 시설/작기의 미관측 미래 평가에서 확인해야 한다.

## 8개 타깃 기준 성능

{table(pd.DataFrame(base_rows))}

단위는 RMSE·MAE에서 개수. ‘유지 대비 개선’ 양수는 개선, 음수는 악화다. seed SD는 RMSE의 표준편차. 지표를 타깃 간 합산하지 않는다.

![기준모델 비교](figures/baseline_comparison.png)

### 네 지표의 해석

- **RMSE:** 큰 오차에 민감하다. 딸기 2화방의 2개 이상 증가 48행은 전체 785행의 6.1%지만 SSE 72.6%를 차지한다. 큰 증가를 놓치는 문제가 RMSE에 드러난다.
- **MAE:** 일반적인 오차 크기를 나타내지만 0·무변화가 많은 전체 평균만 보면 변화 구간의 문제를 감춘다. 딸기 2화방 전체 MAE 0.538개와 변화 구간 MAE 2.439개를 함께 봐야 한다.
- **R²:** 평가 정답의 분산 대비 오차다. 토마토 테스트의 정답 변동이 매우 작아 R²가 -1.70, -0.57, -1.57, -10.12로 나타났다. 산출 오류가 아니라 재계산으로 확인된 값이다. 음수 R²는 테스트 정답 평균을 쓰는 수학적 참조보다 SSE가 크다는 뜻이며, 입력과 정답의 관계가 전혀 없다는 뜻은 아니다. 상수 정답 구간의 R²는 미정의로 남겼다.
- **CCC:** 수량의 변동과 수준 일치를 함께 반영한다. 토마토 0.246~0.394, 딸기 0.720~0.823이다. 딸기에서는 전체 CCC가 높아도 급증 구간의 과소예측이 남아 있으므로 구간 bias와 함께 판단해야 한다.

## 표본 규모와 구성

{table(pd.DataFrame(segment_rows))}

토마토 train은 타깃별 1,497~1,553행, 6시설·8~12작기이며 validation은 2시설, test는 2시설·3작기다. 딸기는 타깃별 train 3,653행·14시설·20작기, validation 780행·3시설·4작기, test 785행·3시설·4작기다. 같은 개체를 반복 조사한 행과 8개 타깃을 독립 표본처럼 합산하면 안 된다.

### 단일 시점 관측 작기 데이터 적격성 및 처리 원칙
- **관측 보존 및 예측 적격성 분리:** 시설 원장 데이터에서 단일 관측일만 존재하는 작기(예: `PF_0020209_01 / crop_sn=81`, 재배기간 2025-09-05~2026-07-15, 실제 꽃수 관측 2025-10-20 1회, 관측 개체 sample_num 2, 4, 8)는 후속 실제 관측 시점이 존재하지 않아 다음 조사값 예측쌍(`target_date` 및 `target`)을 물리적으로 생성할 수 없다.
- **모델링 코호트 제외:** 이러한 데이터는 '원본에서 삭제'하거나 0으로 보간하는 것이 아니며, 원본 관측으로 완전히 보존하되 **다음 관측 예측쌍 생성 불가(`no_next_observation`)로 인하여 4개 토마토 타깃의 학습·검증·평가 모델링 코호트에서 제외**한다. 환경 센서 데이터의 날짜가 존재한다는 이유만으로 꽃수 관측일로 오인 계산하지 않으며, 꽃수 값이 실제로 기록된 날짜만을 관측일로 엄밀히 집계한다.
- `excluded_observations.csv`에 해당 작기의 개체별 제외 사유(`no_next_observation`, 3행)를 명시하여 투명하게 집계한다.

토마토 테스트 양수 표본은 타깃별 9~12행에 불과하다. ‘총 행 수가 충분한가’보다 **새 시설 수·독립 작기 수·양수 및 변화 사건의 수**가 중요한 제한이다. 표본 추가가 성능을 얼마나 높이는지는 현재 결과만으로 확정할 수 없으며, 시설/작기 수를 늘리는 학습곡선을 별도로 검증해야 한다.

## 시설·작기·조사 간격·생육 변화별 진단

### 시설 및 작기

토마토 PF_0026456_01/작기 208은 34~53행으로 테스트의 11.9~16.4%지만 타깃별 SSE의 72.1~91.0%를 차지한다. 다른 시설에서 양호한 평균이 이 시설의 높은 과대예측을 가리고 있다. 토마토 합계 RMSE는 이 시설에서 2.474개, PF_0024697_01에서 0.290개다.

딸기 PF_0024700_01/작기 PF_0024700|7546|2024-09-10의 234행은 2화방 SSE 66.0%, 3화방 68.3%, 합계 64.7%를 차지한다. 합계 RMSE는 4.309개이며 유지 기준 3.927개보다 크다. 이 시설을 제외하면 전체 합계 모델 RMSE가 2.075개가 되지만, **제외해 성능을 개선했다고 보고하지 않는다.** 시설 민감도 확인용이다.

동일 시설의 작기 간 차이도 크다. 토마토 PF_0024697_01의 1화방은 작기 177에서 RMSE 0.159개(N=243), 작기 178에서 0.743개(N=27)다. 딸기 PF_0023732_01의 1화방은 2024 작기 1.569개(N=274), 2025 작기 0.150개(N=81)다. 작은 작기 표본과 수량 분포 차이를 함께 보아야 하며, 특정 시설 특성이나 재배 방식이 원인이라고 단정할 수 없다.

근거: `tables/error_strata.csv`의 facility/crop_cycle 및 `leave_one_facility_sensitivity.csv`.

### 조사 간격

‘직전 조사 후 경과일’은 예측 시점에 아는 입력이고, ‘다음 실제 조사까지 간격’은 사후에 확인되는 예측 거리다. 둘을 분리했다. 모든 작물의 중앙값은 7일이지만 test의 다음 조사 간격은 토마토 최대 10일, 딸기 최대 28일이다.

토마토 3화방은 다음 조사 간격 1~7일 RMSE 0.098개(N=207), 8~14일 0.818개(N=79)다. 다만 이 차이는 시설·작기 초기 및 수량 변화 분포와 겹친다. `crossed_error_strata.csv`에서 시설×경과일×간격을 함께 확인할 수 있다. **간격만 길어서 오차가 증가한다고 인과 해석하지 않는다.** 같은 PF_0026456_01 시설·≤60일로 좁혀도 토마토 합계 RMSE는 1~7일 0.908개(N=27), 8일 이상 5.153개(N=7)로 다르지만, 후자의 표본 7행으로 인과나 일반화를 확정할 수 없다. 딸기 15~28일 구간 10행은 0 중심이라 오히려 오차가 작다. 긴 간격을 일률적으로 고위험 처리할 근거는 없다.

### 작기 경과일 및 수량 변화

토마토는 4개 타깃 모두 작기 시작 후 60일 이내에 SSE의 **99.6~100.0%**가 집중된다. 고정된 1~3화방의 꽃수가 이후 0으로 관측되는 기간이 길다는 구성과 부합한다. ‘60일’을 생리학적으로 확정된 개화 경계로 해석하지 않고, 초기 집중 관측·화방 상태 확인을 위한 후보 구간으로 쓴다.

딸기 1화방은 61~120일에 SSE 74.8%, 2·3화방은 121~180일에 각각 84.6%, 84.4%, 합계는 같은 구간에 65.6%가 집중된다. 이 역시 작기 경과일에 따른 사후 분류이며 실제 생리 단계 라벨이 아니다.

![작기별 오차](figures/crop_age_error.png)

딸기에서 2개 이상 증가한 구간의 평균 bias(예측−실제)는 1·2·3화방·합계 순서로 **-4.49, -4.02, -3.56, -4.77개**다. 반대로 2개 이상 감소 구간에서는 **+2.55, +2.12, +1.45, +2.46개**다. 이는 변화폭을 충분히 따라가지 못하는 양상이다. 수정·적과·수확·낙과 등을 실제 원인으로 확인하려면 시점과 수량이 연결된 작업/상태 기록이 필요하다.

토마토 1화방은 변화 관측뿐 아니라 무변화 관측도 SSE 51.8%를 차지한다. 토마토 전체를 ‘급변만 해결하면 된다’고 일반화하지 않는다. 과대예측이 발생한 초기/시설 구간도 함께 확인해야 한다.

![변화 구간 오차](figures/change_error_concentration.png)

수량 변화 및 양수 여부는 미래 정답으로 나눈 **사후 진단 구간**이다. 운영 화면의 사전 경보나 입력 피처에 그대로 사용할 수 없다. N<30은 소표본 표시, 빈 구간은 0점 대신 미평가로 표시한다.

## 기존 입력 활용 범위와 추가 정보 필요성의 구분

### 확인된 기존 입력 신호

동일 모델·동일 검증 표본에서 범주 하나를 추가한 조합 쌍을 비교했다. 각 조합은 별도 튜닝됐으므로 엄밀한 고정 파라미터 ablation이나 인과 효과가 아니다.

딸기 착과수 관측 이력을 추가한 쌍의 RMSE 차이 중앙값은 1·2·3화방·합계에서 -0.693, -0.586, -0.412, -1.001개다. 해당 비교에서 관측 이력은 유용한 신호다. 토마토의 온습도·CO2 추가 효과는 타깃·조합·모델에 따라 개선과 악화가 함께 나타났다. 예컨대 온습도 추가의 중앙 RMSE 차이는 1화방 +0.009, 2화방 -0.033, 3화방 -0.056, 합계 -0.025개다. ‘환경 정보는 항상 유효/무효’라는 결론은 성립하지 않는다.

### 기존에 있지만 이번 실험에서 미활용한 항목

딸기 prepared 데이터에 `leaves_length`, `leaves_num`, `theca_diameter`가 모두 존재하고, 분석 표본에서 숫자 결측은 없지만 10개 등록 조합에는 포함되지 않았다. **기존 데이터 활용 확대의 우선 후보**다. 관측 단위·스케일·조사 시점의 의미는 원천 정의와 확인해야 하며, 값이 존재한다는 사실만으로 품질 검증을 완료했다고 보지 않는다. 새 AX 수집 효과와 혼동하지 않도록 별도 실험군으로 평가한다.

### 수치 보정·결측의 영향

선정 모델의 입력별 결측률과 학습 범위 이탈률을 `selected_input_diagnostics.csv`에 제공했다. 초기 이력 부족과 아직 양수가 나오지 않아 정의되지 않은 경과일 등이 섞일 수 있으므로 모두 수집 실패로 해석하지 않는다. 토마토 test의 선택 입력은 학습 변수별 최소·최대 범위를 벗어나지 않았지만 시설별 오차는 컸다. 단변량 범위 내라는 사실이 결합 분포의 동일성을 보장하지 않는다.

raw→bounded 보정은 일부 모델에서 자주 발생하지만, 선정 모델의 테스트 RMSE 변화는 최대 약 0.00103개(토마토 합계)다. 현재 성능 차이를 출력 클리핑만으로 설명할 수 없다. 딸기 2화방은 3/785행, 3화방은 1/785행의 정답이 train 상한을 초과한다. 극단 수량 표본 확대와 상한 정책은 향후 별도 민감도 과제로 검토한다.

### 원인 분류 결론

| 구분 | 이번 결과로 확인한 사실 | 아직 확인되지 않은 것 | 다음 검증 |
|---|---|---|---|
| 기존 데이터 활용 한계 | 등록 변수군 범위 제한, 딸기 생육 3항목 미활용, 온습도 추가 효과 혼재 | 기존 데이터 전체가 AI에 부적합하다는 주장 | 동일 표본에서 기존 미활용 변수만 추가 |
| 표본 규모·대표성 | test 시설 2/3개, 토마토 양수 9~12행, 특정 시설·작기에 SSE 집중 | 표본 확대만으로 성능이 얼마나 좋아지는지 | train 시설/작기 단위 학습곡선 및 새 시설 외부평가 |
| 추가 정보 필요성 | 증가·감소를 충분히 추종하지 못함, 작업/화방 상태가 입력에서 직접 식별되지 않음 | 적과·수확·수정 이력이 실제로 얼마의 오차를 줄이는지 | 시간 정합된 이벤트 수집 및 동일 조건의 AX ablation |
| 모델·목표 정의 | 합계 모델은 유지 기준보다 악화, 초기 구간이 전체와 상이 | 더 복잡한 모델 또는 신규 데이터만이 해결책인지 | 동일 데이터의 단순 기준·합계 방식·변화 예측 비교 |

## 타깃별 후속 검증 초점

{table(pd.DataFrame([{'타깃':LABEL[x['target']],'관측 근거':x['finding'],'검증 방향':x['next_validation']} for x in diagnosis]))}

{sources}
'''
write('01_성능_오차_원인분석.md',report1)

hypotheses=[
 dict(id='H01',priority='P0',category='기존 입력 활용',evidence='feature_coverage: 딸기 leaves_length/leaves_num/theca_diameter 숫자 결측 0%, 등록 조합 미포함',hypothesis='잎·관부 상태가 관측 이력만으로 설명되지 않는 착과 변화의 예측에 추가 정보를 줄 수 있음',candidate='기존 잎 길이·엽수·관부 직경',validation='B1 vs B0; 단위·시점 감사 후 train-only 이력 피처 생성',kpi='전체 RMSE 및 증가 구간 MAE',status='추가 효과 미검증'),
 dict(id='H02',priority='P0',category='관측 정의/연결',evidence='토마토 test 0 비율 96.3~97.0%, ≤60일 SSE 99.6~100.0%',hypothesis='화방 상태·0의 의미와 조사 범위가 초기 꽃수 오차를 구분하는 데 도움이 될 수 있음',candidate='화방 ID·발생/개화/종료 단계·0 사유·관측 가능 여부·조사자/근거 사진',validation='원본 라벨 보존, 이중 관측 감사, 동일 audited cohort에서 B2 vs B1',kpi='초기 구간 MAE·양수 구간 RMSE·라벨 불일치율',status='추가 효과 미검증'),
 dict(id='H03',priority='P1',category='행동',evidence='딸기 증가≥2 bias -3.56~-4.49개, 감소≥2 bias +1.45~+2.55개; 토마토 합계 감소≥2 13행 SSE 86.1%',hypothesis='화방 단위 수정·적화·적과·수확·낙과 기록이 수량 변화 방향과 크기 설명에 도움',candidate='작업 유형·시각·적용 범위·제거/수확 수량·수정 방법·작업 전후 상태',validation='forecast 시점까지 실제 사용 가능했던 이력만 A1에 투입; 미래 수행 실적은 사후 원인 확인용',kpi='변화 구간 MAE와 증가/감소 bias',status='원인과 효과 모두 미확정'),
 dict(id='H04',priority='P1',category='맥락',evidence='PF_0026456_01 및 PF_0024700_01의 오차 집중; 단순 수치 범위만으로 설명 불가',hypothesis='시설·품종·재배관리 차이 및 작업 사유가 시설별 편차 일부를 설명할 수 있음',candidate='품종·정식/화방 출현일·목표 착과량·작업 사유 코드·판단 당시 상태',validation='A2 vs A1 동일 새 시설 평가; 시설 ID 암기 대신 실제 맥락 항목 비교',kpi='시설 균등 가중 MAE·worst-facility MAE',status='메타데이터 원천 확인 및 효과 미검증'),
 dict(id='H05',priority='P1',category='표본/조사 설계',evidence='토마토 양수 N 9~12; 조사 간격과 시설·경과일 중첩; 딸기 15~28일 N=10',hypothesis='초기·전환 구간의 반복 조사 및 새로운 시설/작기 확보가 일반화 검증력을 높임',candidate='예정/실제 조사시각·지연 사유·개체 교체·관측 범위·과거 초기 이력',validation='조사 강화 1~2주 PoC와 전체 생육 구간 성능 검증 분리; 시설/작기 단위 학습곡선',kpi='변화 사건 수·독립 시설 수·조사 준수율',status='표본 확대 효과 미검증'),
 dict(id='H06',priority='P2',category='제약',evidence='구간·시설 오차가 있으나 제약 실측 입력 없음; 회의록의 제약 정의 요구',hypothesis='수정·제어·관수·작업 지연을 유발한 제약 기록이 설명력을 추가할 수 있음',candidate='구동기/수분 장치 상태·인력 부족·정전·관수 장애·제약 시작/해제 시각',validation='A3 vs A2; 미발생과 미기록 구분, 당시 알려진 상태만 입력',kpi='제약 노출 구간 MAE 및 전체 비열화',status='현재 자료로 원인 식별 불가'),
 dict(id='H07',priority='P0',category='모델/타깃',evidence='직접 합계 모델 유지 대비 RMSE 토마토 +21.0%, 딸기 +0.54%',hypothesis='직접 합계·개별 예측 합산·변화량 예측 등 설계 선택이 데이터 추가 없이 개선 가능',candidate='신규 수집 없음; 동일 행 교집합과 단순 기준모델',validation='B0 내 설계 비교를 validation에서만 선정; AX arm과 분리',kpi='합계 RMSE·MAE·유지 기준 대비 skill',status='저장된 방식 비교 존재, 미래 성능 확인 필요')]
h=csv('hypotheses',hypotheses)

# Field-level proposal; unavailable outcomes must never be features at the original forecast time.
fields=[]
def field(table_name,name,dtype,unit,source,cadence,join,available,role,required,hypothesis):
 fields.append(dict(table=table_name,field=name,type=dtype,unit=unit,source=source,cadence=cadence,join_key=join,
                    availability_rule=available,use_role=role,required=required,hypothesis=hypothesis))
entity='facility_id + crop_sn + sample_num + truss_id'
for name,dt in [('event_id','string'),('facility_id','string'),('crop_sn','string'),('scope_type','enum: facility/zone/plant/truss'),('sample_num','string nullable'),('truss_id','integer nullable'),('zone_id','string nullable')]:
 field('ax_event',name,dt,'ID','시설/작기/개체 원장','이벤트마다','scope별 적용 범위 매핑','원장 유효기간 일치','join','Y', 'H02-H06')
for name in ['occurred_at','recorded_at','available_at']:
 field('ax_event',name,'timestamp with timezone','Asia/Seoul','작업 로그/서버 접수 기록','이벤트마다',entity,'세 시각 분리; occurred_at<=t AND available_at<=t','asof_filter','Y','H03-H06')
for name,dt,unit,src in [('event_type','enum','표준 코드','작업자/조사원'),('quantity','float nullable','quantity_unit 참조','작업자 계수'),('quantity_unit','enum','개/주/화방/회 등','작업 정의'),('method_code','enum nullable','표준 코드','작업자'),('reason_code','enum nullable','상태/목표/제약 코드','일지+문답'),('reason_text','string nullable','자유 서술','현장 문답'),('evidence_ref','string','이미지/영상/일지 경로','근거 원본'),('record_status','enum','observed_no_event/event_recorded/unknown','일일 확인'),('confidence','enum','verified/unverified/conflict','교차 검증')]:
 field('ax_event',name,dt,unit,src,'발생시 기록·일일 확인',entity,'예측 시점까지 확인·접수된 기록만','historical_feature','Y' if name in ['event_type','quantity_unit','evidence_ref','record_status','confidence'] else '조건부','H03-H04')
for name,dt,unit in [('observed_at','timestamp','Asia/Seoul'),('available_at','timestamp','Asia/Seoul'),('truss_stage','enum','not_emerged/bud/flowering/fruiting/finished/unknown'),('flower_count','integer nullable','개/화방'),('fruit_count','integer nullable','개/화방'),('zero_reason','enum nullable','observed_zero/not_emerged/finished/not_observed/unknown'),('observable','boolean','관측 가능 여부'),('observer_id','string','조사자 ID'),('image_ref','string nullable','근거 이미지'),('protocol_version','string','조사 정의 버전')]:
 field('ax_observation',name,dt,unit,'조사원+시각 근거','정기 조사 및 초기/전환 구간 추가 조사',entity+' + observed_at','관측·가용 시각<=t; t 이후 관측은 정답/사후 설명','state_or_label','Y','H02-H05')
for name,dt,unit in [('cultivar','string','품종'),('transplant_date','date','정식일'),('truss_emergence_date','date nullable','화방 출현일'),('target_fruit_count','integer nullable','목표 개/화방'),('valid_from','timestamp','유효 시작'),('valid_to','timestamp nullable','유효 종료'),('available_at','timestamp','가용 시각')]:
 field('ax_context',name,dt,unit,'농가 원장·현장 확인','작기 시작·변경 발생시',entity,'valid_from<=t<valid_to AND available_at<=t','context_feature','조건부','H04')
for name,dt in [('constraint_type','enum'),('started_at','timestamp'),('resolved_at','timestamp nullable'),('available_at','timestamp'),('severity','enum'),('affected_scope','string')]:
 field('ax_constraint',name,dt,'표준 정의','장치 로그+근거 문답','발생·해제시',entity,'t 당시 알려진 제약만; 미래 해제 시각으로 과거 특성 생성 금지','constraint_feature','조건부','H06')
for name,dt in [('planned_survey_at','timestamp'),('actual_survey_at','timestamp'),('schedule_available_at','timestamp'),('delay_reason','enum nullable'),('replacement_flag','boolean'),('observation_scope','string')]:
 field('survey_schedule',name,dt,'시각/표준 코드','조사 일정·조사원 기록','조사마다',entity,'당시 확정된 계획만 입력; actual/delay는 사후 진단','schedule_or_diagnostic','Y','H05')
for name,dt,unit in [('leaves_length','float','원천 단위 확인'),('leaves_num','integer','매/주 확인'),('theca_diameter','float','원천 단위 확인')]:
 field('existing_growth',name,dt,unit,'기존 생육조사 데이터','기존 조사시',entity.replace(' + truss_id','')+' + observed_at','available_at<=t; unit/scale 감사 후 사용','existing_feature','Y','H01')
# All logical tables carry explicit scope keys; identities never depend on row order.
for table_name in ['ax_observation','ax_context','ax_constraint','survey_schedule','existing_growth']:
    for name,dt in [('record_id','string'),('facility_id','string'),('crop_sn','string'),('sample_num','string nullable'),('truss_id','integer nullable'),('scope_type','enum'),('record_version','string')]:
        field(table_name,name,dt,'ID/버전','유효기간 원장 및 수집 시스템','레코드마다',entity,'scope에 맞는 ID 필수; 부모 원장 FK 검증','join','Y/범위별 조건부','H01-H06')
for name in ['observed_at','available_at']:
    field('existing_growth',name,'timestamp','Asia/Seoul','기존 조사·수집 로그','조사마다',entity,'예측 당시의 가용성 확인; 불명확시 보수적 사용','asof_filter','Y','H01')
csv('ax_field_schema',fields)

report2=f'''# 2. 개선 가설 및 AX 추가 데이터 구축 방향

## 우선순위

**P0: 기존 입력 활용·관측 정의·검증 설계 → P1: 화방 상태와 작업 이력 → P2: 맥락·제약 확장** 순으로 추진한다. 환경·행동·맥락·제약을 한꺼번에 추가해 원인을 구분할 수 없게 만들지 않는다. 추가 정보가 도움이 될 가능성과 실제 확인된 개선 효과를 분리한다.

{table(h[['id','priority','category','evidence','candidate','validation','kpi']])}

이 표의 가설은 현재 오차와 회의 요구에 근거한 **검증 후보**다. 작업 이력이 관측되지 않았으므로 “적과 때문에 오차가 났다” 등으로 확정하지 않는다. H01은 신규 수집이 아니라 기존 자료 활용 과제다. H07은 데이터가 아니라 모델·목표 정의 과제다.

## 수집 범위와 단위

- **상태:** 화방별 출현·개화·착과·종료 상태, 실제 꽃수/착과수, 관측 가능 여부와 0 사유. 기존 생육 변수 3종을 먼저 확인하고 필요한 추가 관측만 수집한다.
- **행동:** 적화·적과·수정·수확·관련 관리 작업의 시점, 적용 범위, 수량 및 단위. 작물별 표준 작업 코드는 발주처/현장 원장과 매핑한다. 미발생·미기록을 구분한다.
- **맥락:** 품종·정식/화방 출현일·목표 착과량·작업 의도와 결정 당시 상태. 시설 ID 자체를 원인 변수의 대용으로 삼지 않는다.
- **제약:** 수정/제어 장치 상태, 인력 부족, 관수·전력 장애 등의 발생·해제와 영향 범위. 실제 해당 작물에서 수집 가능한 항목부터 PoC한다.
- **결과:** 다음 조사 수량·작업 전후 변화·필요시 수확 수량은 별도로 저장한다. 예측 시점 이후 결과는 정답/사후 검증용이고 동일 예측의 입력이 아니다.

필드별 타입·단위·출처·주기·연결 키·가용 시각 규칙을 **`tables/ax_field_schema.csv` ({len(fields)}개 필드 정의)**에 제공했다. 이 스키마는 구축 제안이며 실제 수집 완료 데이터가 아니다. 작업·관측·맥락·제약·일정 테이블은 시설/작기/개체 원장을 참조한다. 행정용 농가 ID와 센서 시설 ID가 다르면 유효기간이 있는 매핑 테이블로 연결한다.

## 데이터 연결 방법

1. 관측 기본 키는 `facility_id + crop_sn + sample_num + truss_id + observed_at`으로 한다. 개체를 교체하면 새로운 plant instance ID 또는 유효기간 매핑을 부여해 이력이 이어지지 않게 한다. 작기 번호를 시설 없이 단독 키로 쓰지 않는다.
2. 작업은 `event_id`를 고유 키로 하고 `scope_type`을 반드시 저장한다. 시설·구역 작업을 임의 개체 작업처럼 복제하지 않는다. 적용 범위 매핑으로 각 개체에 대한 노출 여부를 계산하고, 시설 단위 작업 수량을 개체마다 전량 배분하지 않는다.
3. `occurred_at`(발생), `recorded_at`(기록), `available_at`(시스템에서 사용 가능)을 분리한다. 원래 예측 시점 t에서 입력에 포함하려면 **발생 시각과 가용 시각이 모두 t 이하**여야 한다. 늦게 기록된 과거 사건은 당시 이용 가능했던 입력으로 소급하지 않는다.
4. 센서/작업 집계 창은 기존 규칙과 같은 `[t−1일,t)`, `[t−3일,t)`, `[t−7일,t)`를 기본 후보로 하고 가용 시각도 필터링한다. 결과를 보고 창을 늘리는 경우 새 버전으로 분리한다. 날짜만 있는 기록은 시각을 임의 추정하지 않고 보수적으로 사용 가능 시점을 정의한다.
5. t 이후 실행한 작업은 수량 변화를 설명하는 사후 태그로만 사용한다. t 이전에 확정된 작업 **계획**은 실제 수행 이력과 다른 필드로 저장하고, 계획을 이용한 조건부 예측을 별도 평가한다.
6. 정답은 같은 개체·같은 화방의 다음 관측으로 연결한다. 누락·관측 불가를 0으로 채우지 않는다. 1~3화방 합계는 세 화방의 정답이 모두 있는 경우만 만든다. 0 의미 감사로 라벨을 수정할 때는 원본을 보존하고 수정 사유·버전을 기록한다.
7. 환경은 `facility_id/zone_id + timestamp`로 연결하고 센서 단위·시간대·설치 변경을 기록한다. “해당 주” 문자열만으로 작업과 환경을 붙이지 않는다.

## 현장 PoC와 성능 검증의 분리

**1~2주 PoC는 수집 방법의 실행 가능성 검증**이다. 일지·영상·현장 문답의 기록 일치, 작업 수량 단위, 조사 부담, 지연 기록률을 확인한다. 1~2주만으로 8개 타깃의 예측 개선을 입증했다고 보고하지 않는다. 핵심 생육 구간과 새로운 독립 작기가 포함될 때까지 성능용 수집은 이어가야 한다.

운영 흐름은 다음과 같다.

1. 수집 전: 시설·작기·개체·화방 원장 확정, 표준 작업 코드·단위·0 의미 정의, 예측 시점과 조사 일정 고정.
2. 매일: 사건 발생 기록과 발생하지 않은 날의 확인을 구분. 그림/일지/영상 등 근거 참조를 남기고 충돌은 보류한다.
3. 조사일: 개체·화방 상태와 수량 기록, 가능하면 작업 전후 구분. 초기·전환 구간의 추가 조사 빈도는 현장 부담과 PoC 결과로 조정한다.
4. 주간 검수: 시각 역전, 미연결 키, 중복 이벤트, 결측/0 혼용, 시설 범위 수량의 중복 배분 점검. 오류 기록을 없애지 않고 correction log로 수정한다.
5. PoC 종료: 항목별 수집률·교차 확인 일치율·가용 지연·조사 부담을 보고하고, 유지/수정/중단 항목을 결정한다.

### 수집 품질 KPI 제안

| 항목 | 계산/판정 | 초기 제안 기준 | 성격 |
|---|---|---|---|
| 연결 성공률 | 유효 원장에 연결된 대상 관측·이벤트 / 전체 대상 기록 | ≥99% | 협의 전 제안 |
| 핵심 필드 완결률 | 필수 ID·유형·시각·범위가 있는 기록 / 대상 기록 | ≥95% | 협의 전 제안 |
| 시각 정합성 | 미래 가용 정보 사용·발생/작기 기간 역전 | 예측용 테이블 위반 0건 | 누수 방지 요구 |
| 0/미관측 구분 | unknown 및 관측 불가의 명시 비율 | 구분 필드 100% 제공 | 임의 0 대체 금지 |
| 교차 확인 일치율 | 무작위 감사 표본 중 기록과 근거 일치 | ≥95%, 감사 N·불일치 유형 병기 | 협의 전 제안 |
| 일일 기록 가용성 | 당일/다음 조사 예측 마감 전에 접수된 기록 비율 | 마감 시각을 먼저 확정, PoC에서 지연 분포 산출 | 시간 누수 판단 기준 |

비율의 분모를 수집된 기록으로만 잡아 누락 사건을 숨기지 않는다. 영상/일지/현장 감사로 파악한 대상 사건 수를 가능한 범위에서 별도 집계한다. 실제 발생 사건 전체를 알 수 없을 때는 포착률을 확정하지 않고 감사 범위를 명시한다.

## 실행 과업과 산출물

| 순서 | 과업 | 담당 역할 제안 | 완료 기준 |
|---|---|---|---|
| 1 | 기존 딸기 생육 3항목 단위·시점·개체 키 감사 | 데이터 엔지니어+생육 담당 | 정의서·오류/누락 목록·재현 가능한 연결 표 |
| 2 | 토마토 초기·화방 0 의미 표본 감사 | 생육 담당+현장 조사원 | 양수/0/미관측/종료 사례와 근거, 변경 이력 |
| 3 | 작업·화방 상태 수집 1~2주 PoC | 현장 조사원+데이터 담당 | 이벤트·관측 테이블, 품질 KPI와 수집 부담 |
| 4 | 기존/AX 비교 실험 데이터 동결 | 분석 담당 | 동일 row_id/split/seed 목록, 데이터 가용 시각 검사 |
| 5 | 학습곡선·입력 추가 비교·새 시설/작기 평가 | 모델 담당 | 개선 효과와 불확실성, 부정/무효 결과 포함 |

현재 파일만으로 완료할 수 있는 분석·설계와 향후 현장 수집·학습 실험을 구분했다. 이 문서의 신규 수집 및 미래 비교는 계획이며 수행 결과를 만들어 넣지 않았다.
'''
write('02_AX_추가데이터_구축방향.md',report2)

arms=[
 dict(arm='B0',name='동일 신규 코호트의 기존 입력',inputs='현재 입력 정의',purpose='기존 체계 기준 재학습',comparison='새 코호트 baseline'),
 dict(arm='B1',name='기존 미활용 데이터 확대',inputs='B0 + 감사된 기존 잎·관부 등',purpose='신규 수집 없이 활용 범위 효과',comparison='B1-B0'),
 dict(arm='B2',name='AX 상태·관측 맥락',inputs='B1 + 새 화방 상태/관측 맥락',purpose='새 상태 정보의 증분 효과',comparison='B2-B1'),
 dict(arm='A1',name='AX 행동',inputs='B2 + 가용한 작업 이력',purpose='작업 정보 증분 효과',comparison='A1-B2'),
 dict(arm='A2',name='AX 맥락',inputs='A1 + 결정 이유·목표·추가 맥락',purpose='맥락 증분 효과',comparison='A2-A1'),
 dict(arm='A3',name='AX 제약',inputs='A2 + 당시 알려진 제약',purpose='제약 증분 효과',comparison='A3-A2'),
 dict(arm='ABL',name='개별 추가·제거',inputs='B1+각 AX 범주 단독 및 전체 AX-각 범주',purpose='추가 순서 의존성과 상호작용 확인',comparison='같은 B1 및 전체 대비')]
csv('comparison_arms',arms)
kpis=[]
for t in ORDER:
 x=bt.loc[t]; ch=st(t,'changed_binary','changed')
 kpis.append(dict(target=t,reference_rmse=x.rmse_mean,illustrative_10pct_rmse=x.rmse_mean*.9,
                   reference_change_mae=ch.mae_mean,illustrative_15pct_change_mae=ch.mae_mean*.85,
                   reference_persistence_rmse=x.persistence_rmse,reference_zero_rmse=x.zero_predictor_rmse,
                   status='planning illustration only; rebaseline on the same new paired cohort before preregistration'))
k=csv('kpi_planning_reference',kpis)
report3=f'''# 3. As-Is/To-Be 검증 체계 및 KPI 설정 방향

## 검증 질문

**“같은 예측 시점·같은 정답·같은 시설/작기 분할·같은 학습 예산에서, 추가 입력으로 무엇이 얼마나 개선되는가?”**를 검증한다. 기존 데이터와 신규 AX 데이터의 서로 다른 표본에서 얻은 RMSE를 단순 비교해서 추가 입력 효과라고 부르지 않는다.

현재 동결 결과는 `As-Is v2`의 진단 기준이다. 지금 본 test로 새로운 변수·모델을 고르면 더 이상 독립 확인이 아니다. 새 시설/작기의 미래 관측에서 As-Is와 To-Be를 나란히 재학습·평가한다. 새 데이터에서 기존 입력만 쓴 B0를 반드시 재산출한다.

## 비교 실험군

{table(pd.DataFrame(arms))}

단계적 추가만으로 범주 효과를 확정하면 투입 순서에 의존한다. 사전 선택한 범주 단독 추가 및 전체 입력에서 한 범주 제거를 함께 설계한다. 임의 A/B/C 이름을 실제 변수 범주로 하드코딩하지 않고 실제 정의서와 feature list hash로 실험군을 기록한다.

**라벨 수정 효과를 입력 추가 효과와 분리한다.** 관측 감사로 정답이나 유효 표본이 달라지면 모든 실험군에 같은 새 정답·같은 행을 사용한다. 원본 라벨과 감사 라벨의 차이는 별도 bridge audit로 보고한다. B0와 A1이 서로 다른 정답을 사용하도록 두지 않는다.

## 동일 평가 조건

1. **타깃·시점:** 주 과제는 동일 개체·화방의 다음 조사 수량으로 유지한다. 미래 실제 조사 간격은 사후 진단용이며 t 시점의 입력으로 쓰지 않는다. 고정 7일 과제를 추가하려면 관측 일정·정답 확보 방식부터 별도 프로토콜로 정의하고 임의 보간 라벨로 기존 과제와 혼합하지 않는다.
2. **표본:** 미래 수집 코호트에서 동일 row_id와 타깃별 비교 가능한 표본을 동결한다. AX 결측 때문에 유리한 행만 제거하지 않는다. 공통 관측 부분집합 결과와 전체 운영 코호트의 결측/미기록 대응 결과를 모두 보고한다.
3. **분할:** 미래 성능은 학습 종료 시점 이전 자료만으로 학습하고 이후 관측을 평가한다. 새로운 시설 일반화와 기존 시설의 미래 작기 예측은 별도 결과로 제시한다. 개체/작기/시설 연결로 누수가 발생하지 않게 한다.
4. **전처리:** 중앙값·스케일·clip 상한·인코더를 train에서만 학습한다. 기존 v2와 동일 정책을 주 비교에 사용하고 정책 변경 효과는 별도 ablation으로 분리한다. 각 arm은 자기 입력에 필요한 train-only 통계를 사용하되 절차는 같다.
5. **학습:** 같은 모델 집합·하이퍼파라미터 탐색 예산·교차검증·seeds 42/52/62/72/82를 적용한다. 순수 입력 효과는 고정 모델/같은 튜닝 절차로 먼저 비교하고, 실험군별 최적 모델을 고르는 시스템 비교는 별도 표로 제공한다.
6. **선정:** validation으로 입력군/모델/임계값을 선택해 동결한 뒤 마지막 평가를 한 번 연다. 후보 선택 기록과 feature list·코드·데이터 해시를 저장한다. final holdout를 보고 후보나 KPI를 바꾸면 새로운 holdout가 필요하다.
7. **분석:** 전체·시설 균등 가중·작기·초기/전환·조사 간격·양수/변화 구간을 함께 보고한다. 변화 구간은 사후 결과 분석용이다. 운영상 사전 타깃팅은 예측 시점의 화방 상태와 이력으로 정의하고 validation에서 동결한다.
8. **기준모델:** 현재값 유지, train에서 정한 상수 예측, 0 예측 및 단순 회귀를 포함한다. 0이 많은 타깃에서 전체 RMSE만 낮아지는 방식은 양수/활성 구간 성능과 함께 판정한다.

## KPI 제안과 판정

다음 수치는 **협의·사전등록 전의 제안**이며, 달성 사실이나 발주처 확정 기준이 아니다. 회의록의 RMSE 0.7은 당시 대상·척도·평가 조건을 확인한 뒤에만 적용할 수 있다. 8개 타깃과 합계에 일괄 적용하지 않는다. 회의록의 Recall/F1 언급도 현재 수량 회귀의 주 지표로 그대로 옮기지 않는다. 별도 사건 탐지 과제를 정의할 경우에만 threshold·정답 정의와 함께 보조 지표로 쓴다.

| KPI | 산식/적용 | 초기 제안 | 판정 원칙 |
|---|---|---|---|
| 주 지표: 타깃별 RMSE 개선 | 100×(RMSE_B0−RMSE_AX)/RMSE_B0 | ≥10% 상대 개선 | 같은 새 코호트·같은 seed에서 비교; 효과 불확실성 병기 |
| 증분 AX 효과 | 100×(RMSE_B1−RMSE_AX)/RMSE_B1 | 양의 개선과 실무 유의 크기 확인 | 기존 미활용 데이터 효과와 구분; 범주별 무효/악화도 보고 |
| 핵심 변화 구간 MAE | 실제 변화 관측의 seed별 MAE 평균 | ≥15% 상대 개선 | 사후 진단; 전체 RMSE와 양수/활성 구간 guardrail 병행 |
| 유지/0 기준 비교 | 동일 표본의 기준모델 대비 RMSE 및 활성 구간 MAE | 유지 기준보다 개선; 0 기준보다 나쁜 타깃은 효용 주장 보류 | 0 예측이 운영적으로 충분하다는 뜻 아님 |
| 시설 일반화 | 시설별 MAE의 균등 평균 및 각 시설 변화 | 균등 MAE 개선, 충분한 N의 시설에서 MAE 악화 ≤5% 제안 | 작은 N에서 강제 pass/fail 금지; 시설별 원자료와 함께 판단 |
| 보조 일치도 | R²·CCC, bias, P90 절대오차 | CCC 감소 0.02 이내 제안, 방향별 bias 완화 | 상수 정답에서 R² 미정의; 모든 타깃 R² 0.8 같은 일괄 기준 금지 |
| 데이터 품질 | 02 문서의 연결률·가용 시각·근거 일치율 | 제안 기준 참조 | 성능 KPI와 별도 gate |

### 현재 수치로 보는 목표 크기 예시

{table(pd.DataFrame([{'타깃':LABEL[r.target],'현 기준 RMSE':r.reference_rmse,'10% 개선 예시':r.illustrative_10pct_rmse,'변화 MAE 기준':r.reference_change_mae,'15% 개선 예시':r.illustrative_15pct_change_mae} for r in k.itertuples()]))}

위 숫자는 **현 test의 0.9×RMSE, 0.85×변화 MAE**를 보여주는 기획용 계산이다. 새 코호트의 타깃 분포가 달라지므로 그대로 계약상 절대 통과선으로 삼지 않는다. 토마토는 10% 감소만으로도 0 기준보다 여전히 나쁠 수 있고, 합계는 유지 기준을 넘지 못할 수 있다. 이 경우 효과가 충분하다고 판정하지 않는다.

## 표본 규모와 불확실성 검증

현재 test의 2/3개 시설과 토마토 양수 9~12행으로 모집단 일반화의 정밀한 신뢰구간을 주장하지 않는다. 이번에는 시설 하나씩 제외한 **민감도 범위**를 제공했고, 이를 confidence interval이라 부르지 않았다. seed SD는 같은 표본에서 초기화 차이만 반영한다.

후속 실험은 train 내 시설/작기 블록을 25/50/75/100%로 늘리는 학습곡선을 구성하고, 반복 추출 seed를 고정해 같은 validation에서 평가한다. 동일 개체의 행을 무작위로 쪼개 표본 수를 부풀리지 않는다. 시설 다양성 확대와 기존 시설의 조사 빈도 확대를 별도 축으로 비교한다. 최종 test는 이 설계 선택에 사용하지 않는다.

PoC 후 시설별 오차 차이·변화 사건 발생률·시설 내 상관을 추정해 필요한 독립 시설·작기·사건 수를 정한다. “몇 행이면 충분” 또는 “5시설이면 통계적으로 보장” 같은 숫자를 현재 근거 없이 확정하지 않는다. 미래 개선 차이의 불확실성은 충분한 독립 시설/작기가 확보된 뒤 paired block resampling 등 군집 구조를 보존하는 방식으로 산출한다. 8개 타깃의 동시 개선 주장을 할 경우 다중 비교 계획도 평가 전 문서화한다.

## 실행 순서 및 종료 산출물

- **설계 동결:** 예측 시점·정답·가용 정보·arm별 feature list·주 타깃/KPI·분할을 문서화한다.
- **관측/연결 감사:** H01/H02를 수행하고 원본과 수정 정의의 차이를 보존한다.
- **수집 PoC:** 작업/상태 우선 수집으로 1~2주 품질·부담 검증 후 항목을 조정한다.
- **성능용 코호트 확보:** 초기·전환 구간과 독립 미래 시설/작기가 포함되도록 수집한다.
- **동일 조건 비교:** B0/B1/B2/A1/A2/A3 및 사전 지정 ablation, 학습곡선을 실행한다.
- **최종 판정:** 미관측 평가에서 타깃별 효과·실패·불확실성을 보고하고, 개선이 확인된 항목만 구축 우선순위 근거로 채택한다.

최종 산출물은 데이터 버전/해시, 동일 표본 검증 로그, arm별 지표·시설별 오차, 품질 KPI, 추가 항목별 효과와 비용/부담, 채택·보류·재검증 결정표다. 이번 문서가 이 후속 실행의 기준 설계이며, 신규 AX의 성능 개선 효과가 이미 확인됐다는 의미는 아니다.
'''
write('03_AsIs_ToBe_검증체계_KPI.md',report3)

# Workbook is useful for review; full prediction rows remain parquet.
frames={p.stem[:31]:pd.read_csv(p) for p in sorted(T.glob('*.csv')) if p.stem!='integrity_checks'}
frames={'README':pd.DataFrame([{'항목':'기준','내용':'2026-09-14 완료 실험 진단; holdout 재사용; 새 AX 효과 미검증'},
 {'항목':'집계','내용':'지표는 seed별 계산 후 평균; SD는 초기화 변동; row n은 seed 반복 제외'},
 {'항목':'KPI','내용':'모든 신규 목표 수치는 협의 전 제안; 현재 달성 아님'},
 {'항목':'행별 예측','내용':'tables/selected_predictions.parquet; target_id=과제, target=실제 개수'},
 {'항목':'범주 비교','내용':'validation에서 모델 고정·각 조합 별도 튜닝; 인과 효과 아님'}]),**frames}
write_xlsx(HERE/'분석표_및_AX설계.xlsx',frames)

# Offline HTML: embedded figures and filterable computed strata, without external dependencies.
import base64
parts=[]
for name in ['01_성능_오차_원인분석.md','02_AX_추가데이터_구축방향.md','03_AsIs_ToBe_검증체계_KPI.md']:
 text=(HERE/name).read_text()
 for p in F.glob('*.png'):text=text.replace('figures/'+p.name,'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
 parts.append(markdown.markdown(text,extensions=['tables','fenced_code','toc']))
records=json.loads(s.where(pd.notna(s),None).to_json(orient='records',force_ascii=False))
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>생육 예측 성능 원인 분석 및 AX 구축 방향</title><style>
body{font-family:system-ui,'NanumGothic',sans-serif;line-height:1.8;color:#22323d;background:#f2f5f5;margin:0}main{max-width:1180px;margin:auto;background:white;padding:40px}h1{font-size:28px;border-bottom:3px solid #136f63;padding-bottom:16px}h2{margin-top:42px;color:#136f63}h3{margin-top:28px}table{border-collapse:collapse;display:block;overflow:auto;width:100%;font-size:13px;margin:18px 0}th,td{border:1px solid #d6e1e1;padding:8px;min-width:72px}th{background:#e7f0ef}img{max-width:100%}code{background:#edf1f4;padding:2px 5px;overflow-wrap:anywhere}a{color:#136f63}select{padding:10px;margin:6px;border:1px solid #a5b9b6}section{margin-bottom:60px}.note{background:#fff4da;padding:18px}nav{padding:18px;background:#e7f0ef}@media print{main{padding:5px}h1{break-before:page}table{font-size:10px}select{display:none}}
</style><main><nav><strong>분석 산출물 · 2026-09-14</strong><br>원본 실험 완료: 2026-09-12 · 8개 타깃 · 39,856회<br><a href="#explore">구간별 분석표 탐색</a> · <a href="분석표_및_AX설계.xlsx">Excel 분석표</a></nav>'''
html+=''.join('<section>'+p+'</section>' for p in parts)
html+='''<section id="explore"><h1>구간별 분석표 탐색</h1><p class="note">검증으로 확정한 모델의 bounded 결과입니다. 미래 정답의 변화 여부로 나눈 구간은 사후 진단이며, 현재 holdout는 독립 재검증이 아닙니다. 소표본 N&lt;30은 별도로 표시합니다.</p><select id="target"></select><select id="split"><option value="test">test (진단용)</option><option value="validation">validation (선정에 사용)</option></select><select id="dimension"></select><div id="results"></div></section></main><script>'''
html+='const rows='+json.dumps(records,ensure_ascii=False,allow_nan=False)+';const labels='+json.dumps(LABEL,ensure_ascii=False)+';'
html+='''const target=document.getElementById('target'),split=document.getElementById('split'),dimension=document.getElementById('dimension');
for(const [k,v] of Object.entries(labels)){const o=new Option(v,k);target.add(o)}
const dims={all:'전체',facility:'시설',crop_cycle:'작기',horizon_days:'다음 실제 조사 간격(사후)',previous_gap_days:'직전 조사 간격',crop_age_days:'작기 경과일',change:'수량 변화 크기(사후)',changed_binary:'변화 여부(사후)',label_positive:'정답 양수 여부(사후)',current_positive:'현재 수량 양수 여부',history_count:'관측 이력 수',selected_input_missing:'선정 입력 결측',selected_input_outside_train:'선정 입력 train 범위 이탈'};
for(const[k,v]of Object.entries(dims))dimension.add(new Option(v,k));
function fmt(x){return x===null||x===undefined?'미정의':Number(x).toFixed(3)}
function render(){const selected=rows.filter(r=>r.target===target.value&&r.split===split.value&&r.dimension===dimension.value);const tab=document.createElement('table');const head=tab.insertRow();['구간','N','RMSE','MAE','R²','CCC','bias','SSE 비중 %','주의'].forEach(v=>{const c=document.createElement('th');c.textContent=v;head.appendChild(c)});for(const r of selected){const tr=tab.insertRow();[r.level,r.n,fmt(r.rmse_mean),fmt(r.mae_mean),fmt(r.r2_mean),fmt(r.ccc_mean),fmt(r.bias_mean),fmt(100*r.squared_error_share),r.small_n?'소표본':''].forEach(v=>tr.insertCell().textContent=v)}document.getElementById('results').replaceChildren(tab)}
[target,split,dimension].forEach(e=>e.addEventListener('change',render));render();</script></html>'''
write('종합분석보고서.html',html)
print('Built reports, figures, workbook and HTML',flush=True)
