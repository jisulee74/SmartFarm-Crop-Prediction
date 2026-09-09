from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from .core import digest

AX=Path(__file__).resolve().parents[1]
STRAW=AX/'strawberry_fruit_set_prediction'
SCHEMA=STRAW/'data_audits/fruit_zero_semantics_20260812/artifacts/schema_columns.csv'
META=STRAW/'data/sensor_category_cache/sensor_metadata.csv'


def classify(section,code):
    if section=='CR':
        if code.startswith(('CC01_','CC03_')): return 'Action','환경 제어','창 작동정보'
        for codes,name in [({'CC0403','CC0503','CC3101','CC3201'},'커튼 작동정보'),({'CC0903','CC1203','CC23_C_C','CC23_H_C'},'냉난방 작동정보'),({'CC0803','CC1803','CC1903'},'공기순환 작동정보'),({'CC22_H_C','CC22_D_C'},'가습·제습 작동정보'),({'CC1403','CC2503'},'CO2 공급 작동정보'),({'CC1103'},'보광 작동정보'),({'CC2003','CC2103','CC2603','CC2703','CC2803','CC2903'},'관수·관비 작동정보')]:
            if code in codes:return 'Action','환경 제어',name
        return 'Action','환경 제어','기타 장치 작동정보'
    if section=='EI':
        return 'State','실내환경',('실내 온습도' if code in {'TI','HI','HI01','HI02','DT'} else '실내 CO2' if code=='CI' else '실내 바람' if code=='WS' else '실내 광환경')
    if section=='EO':
        return 'State','외부환경',('외부 온습도' if code in {'TE','HE','HR','DT','DC'} else '외부 바람' if code in {'WS','WD'} else '강우·적설' if code in {'RF','RP','CS','DS'} else '외부 광환경')
    return 'State','근권·양액',('근권·배지 상태' if code in {'RT','TL','CL','AT','HL','BW'} else '양액·배액 상태')


def category_id(role,domain,name): return 'CAT_'+digest([role,domain,name])[:12]+'_V1'


def build_catalog():
    schema=pd.read_csv(SCHEMA).fillna('');meta=pd.read_csv(META).fillna('')
    sources=[]; variables=[]; categories={}; members=[]
    def add_category(role,domain,name):
        cid=category_id(role,domain,name)
        categories[cid]={'Category_ID':cid,'AX_Category':role,'Variable_Domain':domain,'Variable_Category':name,'Version':1,'Status':'candidate_not_frozen'}
        return cid
    def add(f,name,table,field,cid,derivation,window,unit,level,status='definition_ready',reason=''):
        variables.append(dict(Variable_ID=f,Variable_Name=name,Source_Table=table,Source_Field_or_Code=field,Category_ID=cid,Derivation=derivation,Window=window,Unit=unit,Entity_Level=level,Reference_Time='feature_date',Missing_Rule='train median after pre-freeze screening; no row deletion',Definition_Version=1,Availability_Status=status,Reason=reason))
        members.append({'Category_ID':cid,'Variable_ID':f})
    for r in schema.itertuples():
        sources.append({'Source_Table':r.table_name,'Source_Field_or_Code':r.column_name,'Description':r.column_comment,'Data_Type':r.column_type,'Evidence':'local_schema_snapshot','Status':'schema_only'})
        # Identifiers, target-only and ambiguous columns stay in source inventory.
        growth={'leaves_length':('잎 생육정보','cm'),'leaves_width':('잎 생육정보','cm'),'leaves_num':('잎 생육정보','개'),'petiole_length':('잎 생육정보','cm'),'stem_diameter':('줄기·관부 생육정보','mm'),'theca_diameter':('줄기·관부 생육정보','mm'),'grow_length':('줄기·관부 생육정보','cm'),'fruit_cluster_num':('화방 관측정보','개'),'fruits_num':('과실 생육정보','개'),'fruits_weight':('과실 생육정보','g')}
        if r.table_name=='sfkr_pvsn_grow' and r.column_name in growth:
            name,unit=growth[r.column_name];cid=add_category('State','작물 생육',name)
            for suffix,der,w in growth_definitions():add(r.column_name+'__'+suffix,str(r.column_comment)+' '+suffix,r.table_name,r.column_name,cid,der,w,'source unit requires verification' if unit!='개' else unit,'individual','needs_review','measurement unit/provenance must be verified' if unit!='개' else 'raw history required')
        elif r.table_name=='sfkr_pvsn_crop' and r.column_name in {'cropping_system','item_code','cultivation_area','cal_cultivation_area','plant_num','cal_plant_num','plant_density'}:
            cid=add_category('Context','재배 맥락','재배방식' if r.column_name in {'cropping_system','item_code'} else '재배규모·밀도')
            add(r.column_name,str(r.column_comment),r.table_name,r.column_name,cid,'current as-of feature date','current','source definition','crop_cycle','needs_review','as-of validity and units require verification')
    for base,name,table,field,cat,level in [('flower_count','전체 꽃 수','sfkr_hbfm_grow','growth_measure_code + resolved value column','꽃 관측정보','individual'),('fruit_count','해당 화방 착과수','sfkr_pvsn_grow','first_fruits_num / second_fruits_num / third_fruits_num','착과 관측정보','truss'),('current_max_truss_number','최대 관측 화방번호','sfkr_hbfm_grow','growth_measure_code','화방 관측정보','individual'),('positive_truss_count','꽃이 있는 화방 수','sfkr_hbfm_grow','growth_measure_code + resolved value column','화방 관측정보','individual'),('observed_truss_count','관측 화방 수','sfkr_hbfm_grow','growth_measure_code','관측범위·품질','individual')]:
        cid=add_category('Context' if cat=='관측범위·품질' else 'State','작물 생육',cat)
        for suffix,der,w in growth_definitions():add(base+'__'+suffix,name+' '+suffix,table,field,cid,der,w,'개' if 'number' not in base else '번호',level)
    for r in meta.itertuples():
        cid=add_category(*classify(r.sect_code,r.com_code)); binary=str(r.is_binary).lower()=='true'
        sources.append({'Source_Table':'sfkr_hbfm_env_con','Source_Field_or_Code':r.variable_id,'Description':r.com_name,'Data_Type':'binary' if binary else 'numeric','Evidence':'local_sensor_code_dictionary','Status':'code_only'})
        cumulative='누적' in r.com_name or '총급' in r.com_name or '총배' in r.com_name
        ambiguous=r.com_code in {'WD','LW','ID'} or cumulative or r.sect_code=='NT'
        for w in (1,3,7):
            for stat in (('increase',) if cumulative else ('active_fraction',) if binary else ('mean','std','min','max')):
                add(f'{r.variable_id}__{w}d_{stat}',f'{r.com_name} 직전 {w}일 {stat}','sfkr_hbfm_env_con',f'{r.sect_code}/{r.com_code} -> sen_val',cid,stat+('; ddof=0' if stat=='std' else ''),f'[t-{w}d,t)',r.unit,'facility','needs_review' if ambiguous else 'definition_ready','reset/unit/circular-value semantics require verification' if ambiguous else '')
    for f,name in [('days_since_previous','이전 조사 후 경과일'),('days_since_first_measurement','최초 조사 후 경과일'),('days_since_crop_start','작기 시작 후 경과일'),('days_since_first_positive','최초 양수 관측 후 경과일'),('history_count','누적 조사 횟수'),('week_sin','연중 주차 사인값'),('week_cos','연중 주차 코사인값')]:
        cid=add_category('Context','재배 맥락','조사이력' if f=='history_count' else '시간정보')
        add(f,name,'growth/crop source','measurement date/cropping_date/current count',cid,'observed history only; ISO week period=52.1775','through current','count' if f=='history_count' else 'unitless' if f.startswith('week') else 'day','individual')
    return pd.DataFrame(sources),pd.DataFrame(variables),pd.DataFrame(categories.values()),pd.DataFrame(members)


def growth_definitions():
    # Order gives current and delta priority during affine-dependence screening.
    yield 'current','current valid observation','current'
    yield 'delta','current minus previous','1 survey'
    yield 'delta_per_day','delta / positive elapsed days','1 survey'
    for lag in (1,2,3):yield f'lag{lag}',f'lag {lag}',f'{lag} surveys'
    for w in (2,3,4):
        for stat in ('mean','std','min','max','slope'):
            yield f'roll{w}_{stat}',stat+('; ddof=0' if stat=='std' else '; per calendar day' if stat=='slope' else ''),f'{w} valid surveys including current; min_periods={w}'


def snapshot_db(output):
    """Only metadata SELECTs, transaction explicitly read-only; never exports credentials."""
    if not os.getenv('FARMSTOM_DB_PASSWORD'):return {'status':'blocked','reason':'FARMSTOM_DB_PASSWORD missing','db_inventory_complete':False}
    from AXData.tomato_flower_count_prediction.extract import connect
    conn=connect()
    try:
        with conn.cursor() as c:
            c.execute('SET TRANSACTION READ ONLY');c.execute('START TRANSACTION')
            c.execute('SELECT table_name,column_name,column_type,column_comment FROM information_schema.columns WHERE table_schema=DATABASE() ORDER BY table_name,ordinal_position')
            columns=pd.DataFrame(c.fetchall(),columns=['table_name','column_name','column_type','column_comment'])
        output.mkdir(parents=True,exist_ok=True);columns.to_csv(output/'db_schema.csv',index=False,encoding='utf-8-sig')
        return {'status':'schema_collected','db_inventory_complete':False,'reason':'value coverage and code-table review still required','columns':len(columns)}
    finally:conn.rollback();conn.close()
