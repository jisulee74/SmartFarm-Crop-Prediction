"""Export eight observed count totals by actual target timestamp and facility split."""
import json
from pathlib import Path
import pandas as pd
from .catalog import AX



def match_strawberry_dates(frame,source):
    frame=frame.copy()
    source=source.copy()
    source['item_code']=pd.to_numeric(source.item_code,errors='coerce')
    for col in ['cropping_date','cropping_end_date']:
        source[col]=pd.to_datetime(source[col],errors='coerce').dt.normalize()
    records=[]
    for row in frame[['facility_id','crop_sn','timestamp']].drop_duplicates().itertuples(index=False):
        user='_'.join(row.facility_id.split('_')[:2])
        candidates=source[source.user_id.eq(user)&source.item_code.eq(80400)&source.cropping_date.le(row.timestamp)&source.cropping_end_date.ge(row.timestamp)].drop_duplicates()
        record={'facility_id':row.facility_id,'crop_sn':row.crop_sn,'timestamp':row.timestamp,'cropping_number':pd.NA,'matched_crop_sn':pd.NA,'cropping_serl_no':pd.NA,'crop_start_date':pd.NaT,'crop_end_date':pd.NaT,'crop_match_count':len(candidates),'crop_match_method':'user_item_observation_date','crop_match_status':'unmatched' if len(candidates)==0 else 'ambiguous' if len(candidates)>1 else 'matched'}
        if len(candidates)==1:
            c=candidates.iloc[0]
            record.update(matched_crop_sn=c.crop_sn,cropping_serl_no=c.cropping_serl_no,crop_start_date=c.cropping_date,crop_end_date=c.cropping_end_date)
        records.append(record)
    return frame.merge(pd.DataFrame(records),on=['facility_id','crop_sn','timestamp'],validate='many_to_one')


def attach_cycles(frame, cycles, keys):
    cycles=cycles.copy()
    for c in ['cropping_date','cropping_end_date']:
        cycles[c]=pd.to_datetime(cycles[c],errors='raise').dt.normalize()
    cycles=cycles.drop_duplicates()
    if cycles.duplicated(keys).any():raise ValueError('Conflicting crop dates for same cycle')
    merged=frame.merge(cycles,on=keys,how='left',validate='many_to_one',indicator=True)
    if not merged['_merge'].eq('both').all():raise ValueError('Unmatched crop cycle')
    if merged[['cropping_date','cropping_end_date']].isna().any().any():raise ValueError('Missing source crop dates')
    if not ((merged.timestamp>=merged.cropping_date)&(merged.timestamp<=merged.cropping_end_date)).all():raise ValueError('Observation outside recorded crop cycle')
    return merged.drop(columns='_merge').rename(columns={'cropping_date':'crop_start_date','cropping_end_date':'crop_end_date'})


def cycle_summary(frame):
    records=[]
    for (timestamp,split),g in frame.groupby(['timestamp','split']):
        c=g[['facility_id','crop_sn','crop_start_date','crop_end_date']].drop_duplicates()
        records.append({'timestamp':timestamp,'split':split,'crop_cycle_count':len(c),
            'crop_start_date':c.crop_start_date.iloc[0] if len(c)==1 else pd.NaT,
            'crop_end_date':c.crop_end_date.iloc[0] if len(c)==1 else pd.NaT,
            'earliest_crop_start_date':c.crop_start_date.min(),'latest_crop_end_date':c.crop_end_date.max()})
    return pd.DataFrame(records)


def export_cycle_detail(frame,columns,path):
    keys=['timestamp','split','facility_id','crop_sn','sample_num','cropping_number','cropping_serl_no','crop_start_date','crop_end_date','matched_crop_sn','crop_match_method','crop_match_status','crop_match_count']
    entity_keys=['timestamp','facility_id','crop_sn','sample_num']
    if frame.duplicated(entity_keys).any():raise ValueError('Duplicate individual observation')
    detail=frame[keys+columns].sort_values(entity_keys,kind='stable').reset_index(drop=True)
    for c in columns:detail[c]=detail[c].astype('Int64')
    original=frame.groupby(['timestamp','split'])[columns].sum(min_count=1)
    rebuilt=detail.groupby(['timestamp','split'])[columns].sum(min_count=1)
    pd.testing.assert_frame_equal(original,rebuilt,check_dtype=False)
    detail=detail.drop(columns=['cropping_number','crop_match_method','crop_match_status','crop_match_count'])
    detail.to_csv(path,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')


def export(output=None):
    root=Path(output) if output else Path(__file__).parent/'outputs'
    prepared=root/'prepared'
    def load(target):
        return pd.concat([pd.read_parquet(prepared/target/f'{s}.parquet').assign(split=s) for s in ['train','validation','test']],ignore_index=True)
    tomato=load('tomato_total_flower')
    raw=pd.read_parquet(AX/'tomato_flower_count_prediction/data/raw/growth.parquet')
    keys=['facility_id','crop_sn','sample_num']
    for k in keys:
        raw[k]=raw[k].astype(str);tomato[k]=tomato[k].astype(str)
    tomato_cycles=raw[['facility_id','crop_sn','cropping_serl_no','cropping_date','cropping_end_date']].drop_duplicates()
    raw['timestamp']=pd.to_datetime(raw.examin_date).dt.normalize()
    raw['truss']=pd.to_numeric(raw.growth_measure_code)-10000234+1
    raw['value']=pd.to_numeric(raw.flower_count,errors='coerce')
    raw=raw[raw.truss.isin([1,2,3]) & raw.value.ge(0)].copy()
    rkeys=keys+['timestamp','truss']
    # Repeated identical source/crop rows do not add flowers; conflicting values fail.
    requested=tomato[keys+['target_date']].rename(columns={'target_date':'timestamp'})
    raw=raw.merge(requested.drop_duplicates(),on=keys+['timestamp'],validate='many_to_one')
    if raw.groupby(rkeys).value.nunique().gt(1).any():raise ValueError('Conflicting tomato truss counts at target dates')
    raw=raw.drop_duplicates(rkeys)
    wide=raw.pivot(index=keys+['timestamp'],columns='truss',values='value').reindex(columns=[1,2,3])
    wide.columns=['tomato_flower_truss1','tomato_flower_truss2','tomato_flower_truss3']
    t=tomato.rename(columns={'target_date':'timestamp','target':'tomato_flower_total'}).merge(wide.reset_index(),on=keys+['timestamp'],how='left',validate='one_to_one')
    tc=['tomato_flower_truss1','tomato_flower_truss2','tomato_flower_truss3','tomato_flower_total']
    # Totals refer to all observed trusses, not only the first three.
    ts=t.groupby(['timestamp','split'])[tc].sum(min_count=1)
    counts=t.groupby(['timestamp','split'])[tc].count().add_suffix('_n_observed')
    parts=[]
    for number,name in enumerate(['first','second','third'],1):
        d=load('strawberry_'+name)
        col=f'strawberry_fruit_truss{number}'
        parts.append(d[['row_id','facility_id','crop_sn','sample_num','target_date','split','target']].drop(columns='row_id').rename(columns={'target_date':'timestamp','target':col}))
    skeys=['facility_id','crop_sn','sample_num','timestamp','split']
    s=parts[0]
    for part in parts[1:]:s=s.merge(part,on=skeys,how='outer',validate='one_to_one')
    sc=[f'strawberry_fruit_truss{i}' for i in [1,2,3]]
    if s[sc].isna().any().any():raise ValueError('Strawberry truss evaluation cohorts do not align')
    s['strawberry_fruit_total']=s[sc].sum(axis=1,min_count=3);sc+=['strawberry_fruit_total']
    source_path=AX/'strawberry_fruit_set_prediction/data_audits/fruit_zero_semantics_20260812/artifacts/crop_cycles_source.csv'
    source=pd.read_csv(source_path,dtype=str)
    tomato_cycles=tomato_cycles.merge(t[['facility_id','crop_sn']].drop_duplicates(),on=['facility_id','crop_sn'],validate='one_to_one')
    # Original cropping_number was not retained: do not invent a direct match.
    t=attach_cycles(t,tomato_cycles,['facility_id','crop_sn'])
    t['cropping_number']=pd.NA
    t['matched_crop_sn']=pd.NA
    t['crop_match_method']='legacy_cached_date_assignment'
    t['crop_match_status']='direct_match_pending_missing_cropping_number'
    t['crop_match_count']=pd.NA
    direct_path=root/'source_snapshot/tomato_direct_crop_dates.parquet'
    if direct_path.exists():
        direct=pd.read_parquet(direct_path)
        direct['crop_sn']=direct.crop_sn.astype(str)
        direct['timestamp']=pd.to_datetime(direct.timestamp)
        join_keys=['facility_id','crop_sn','timestamp']
        if direct.duplicated(join_keys).any():raise ValueError('Ambiguous direct crop dates')
        replace=['cropping_number','cropping_serl_no','matched_crop_sn','crop_start_date','crop_end_date','crop_match_method','crop_match_status','crop_match_count']
        t=t.drop(columns=replace).merge(direct[join_keys+replace],on=join_keys,how='left',validate='many_to_one')
        if not t.crop_match_status.eq('matched').all():raise ValueError('Missing direct crop match at evaluation date')
    s=match_strawberry_dates(s,source)
    ss=s.groupby(['timestamp','split'])[sc].sum(min_count=1)
    result=ts.join(ss,how='outer').reset_index()
    result['_order']=result.split.map({'train':0,'validation':1,'test':2})
    result=result.sort_values(['timestamp','_order']).drop(columns='_order')
    if result.duplicated(['timestamp','split']).any():raise ValueError('Duplicate output keys')
    for c in tc+sc:result[c]=result[c].astype('Int64')
    out=root/'timeseries';out.mkdir(exist_ok=True)
    path=out/'observed_counts_8_by_timestamp_split.csv'
    result.to_csv(path,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    for crop,columns in [('tomato',tc),('strawberry',sc)]:
        crop_frame=result[['timestamp','split',*columns]].dropna(subset=columns,how='all')
        detail=t if crop=='tomato' else s
        crop_frame=crop_frame.merge(cycle_summary(detail),on=['timestamp','split'],validate='one_to_one')
        export_cycle_detail(detail,columns,out/f'{crop}_observed_counts_by_crop_cycle.csv')
        crop_frame.to_csv(out/f'{crop}_observed_counts_by_timestamp_split.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    counts.join(s.groupby(['timestamp','split'])[sc].count().add_suffix('_n_observed'),how='outer').reset_index().to_csv(out/'observation_counts.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    # Check every exported total against the exact prepared target values.
    for target,col in [('tomato_total_flower','tomato_flower_total'),('strawberry_first','strawberry_fruit_truss1'),('strawberry_second','strawberry_fruit_truss2'),('strawberry_third','strawberry_fruit_truss3')]:
        expected=load(target).groupby(['target_date','split']).target.sum()
        actual=result.set_index(['timestamp','split'])[col].reindex(expected.index)
        assert (actual.to_numpy(dtype=float)==expected.to_numpy()).all(),target
    note='''# 날짜·분할별 실제 관측 합계\n\n- observed_counts_8_by_timestamp_split.csv: timestamp, split, 8개 관측값 열.\n- timestamp는 최신 ax_catalog_experiments 실험의 target_date(다음 조사일)이며, feature_date나 예측값이 아닙니다.\n- 원래 시설 기준 train/validation/test 배정을 그대로 사용합니다.\n- 각 값은 그 날짜·split에서 실제 평가 대상으로 사용된 관측 개체의 합입니다. 시설 전체 생산량이나 모든 작물 개체의 총량은 아닙니다.\n- tomato_flower_truss1/2/3: 개체별 해당 화방의 유효 원천 꽃 수를 합산.\n- tomato_flower_total: 기존 전체 꽃 수 타깃(전체 유효 관측 화방 합)의 개체 간 합. 1~3화방만의 합이 아닙니다.\n- strawberry_fruit_truss1/2/3: 해당 화방의 실제 착과수 합. strawberry_fruit_total은 세 화방의 합.\n- 조사 없음/화방 미관측은 빈칸이며 실제 0과 구별합니다. 일부 개체만 해당 화방이 관측된 경우 합계는 유효 관측분의 합입니다.\n- 날짜마다 관측 개체 수가 다릅니다. 합계 증가를 개체당 생육 증가로 해석하지 마세요. observation_counts.csv에 값별 유효 관측 개체 수를 함께 제공합니다.\n- 토마토·딸기는 서로 다른 시설·개체입니다. 같은 행은 날짜와 split만 공유합니다.\n- 학습/검증/테스트 어느 쪽에도 없는 원천 행은 포함하지 않습니다.\n\n재생성: `envs/jslee_py311/bin/python -m AXData.ax_catalog_experiments.export_timeseries`\n'''
    note+='\n작물별 파일: tomato_observed_counts_by_timestamp_split.csv, strawberry_observed_counts_by_timestamp_split.csv. 각 파일은 timestamp, split, 해당 작물의 4개 값으로 구성되며 해당 작물 관측이 전혀 없는 행은 제외합니다.\n'
    note+='\n作期日期 / 작기 날짜: 원천 DB 추출본의 cropping_date, cropping_end_date를 사용합니다. 실시간 DB 재조회가 아니며 DB 종료일이 실제 종료 확정일인지 예정일인지는 별도 확인이 필요합니다. 단일 작기 행은 crop_start_date/crop_end_date에 날짜를 표시하고, 여러 작기가 섞인 행은 두 열을 빈칸으로 둡니다. crop_cycle_count와 earliest_crop_start_date/latest_crop_end_date는 집계 범위입니다. 작기별 정확한 날짜는 *_observed_counts_by_crop_cycle.csv에서 시설·작기별로 확인하세요.\n'
    note+='\n현재 작기 연결: 딸기는 sfkr_pvsn_crop 추출본의 user_id + item_code(080400) + 조사일 포함으로 재매칭하며 1개만 날짜를 확정합니다. 0개/복수 매칭은 날짜 빈칸과 unmatched/ambiguous로 보존합니다. matched_crop_sn은 작기 테이블 PK이고 기존 crop_sn은 실험 추적용입니다. 토마토는 원천 cropping_number 미보존 및 DB 연결 부재로 직접 연결 미완료이며 기존 날짜를 보존하고 direct_match_pending_missing_cropping_number로 표시합니다. cropping_serl_no를 cropping_number로 복제하지 않습니다.\n'
    if direct_path.exists():
        note+='\n업데이트: 토마토 날짜는 source_snapshot/tomato_direct_crop_dates.parquet의 실시간 DB 직접 매칭 결과를 사용합니다. 위의 토마토 미연결 설명은 직접 매칭 캐시가 없는 경우에만 적용됩니다. crop_sn은 기존 실험 추적용이며 matched_crop_sn은 직접 연결된 sfkr_pvsn_crop.sn입니다.\n'
    note+='\n개체별 상세 파일: *_observed_counts_by_crop_cycle.csv는 timestamp + split + facility_id + crop_sn + sample_num별 한 행입니다. 개체 간 합산하지 않습니다. 각 값은 해당 개체의 실제 관측값이며 전체 값만 해당 개체의 화방 간 합계입니다. timestamp_split 파일은 기존 날짜·분할 합계입니다. 상세 파일에서는 cropping_number, crop_match_method, crop_match_status, crop_match_count 열을 내보내지 않습니다.\n'
    (out/'README.md').write_text(note)
    print(json.dumps({'path':str(path),'rows':len(result),'columns':list(result),'split_rows':result.groupby('split').size().to_dict(),'totals_verified':True},ensure_ascii=False,indent=2))

if __name__=='__main__':export()
