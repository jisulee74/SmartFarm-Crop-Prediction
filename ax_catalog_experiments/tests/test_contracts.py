import json
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest
from AXData.ax_catalog_experiments.core import freeze,union,qualify,growth_features,sensor_features,facility_split,initial_groups,neighbours
from AXData.ax_catalog_experiments.catalog import build_catalog
from AXData.ax_catalog_experiments.report import write_xlsx


def test_frozen_members_and_identity(tmp_path):
    p=tmp_path/'f.json';freeze(p,{'a':['x','y']});freeze(p,{'a':['x','y']})
    with pytest.raises(ValueError):freeze(p,{'a':['x']})
    assert union({'a':['x','y'],'b':['y','z']},['a','b'])==['x','y','z']
    with pytest.raises(ValueError):union({'a':['x']},['b'])


def test_train_only_screen_affine():
    train=pd.DataFrame({'current':[2,3,7,8,9],'delta':[1,2,1,2,2]})
    train['previous']=train.current-train.delta;train['constant']=1
    kept,audit=qualify(train,['current','delta','previous','constant','absent'])
    assert kept==['current','delta']
    assert audit[2]['Status']=='affine_dependent_on_prior_members'
    assert audit[-1]['Status']=='missing_over_50_percent'


def test_growth_calendar_slope_and_future_invariance():
    d=pd.DataFrame({'id':['a']*4,'feature_date':pd.to_datetime(['2025-01-01','2025-01-03','2025-01-06','2025-01-07']),'x':[2,6,12,100]})
    a=growth_features(d,['id'],['x'])
    assert a.iloc[2].x__roll3_slope==pytest.approx(2)
    assert a.iloc[1].x__roll2_std==2
    assert a.iloc[0].x__roll2_mean!=a.iloc[0].x__roll2_mean
    d.loc[3,'x']=99999
    b=growth_features(d,['id'],['x'])
    pd.testing.assert_frame_equal(a.iloc[:3],b.iloc[:3])


def test_sensor_half_open_binary_invalid_and_future():
    rows=pd.DataFrame({'facility_id':['f'],'feature_date':pd.to_datetime(['2025-01-08'])})
    sensor=pd.DataFrame({'facility_id':['f']*4,'variable_id':['EI__TI']*4,'meas_date':pd.to_datetime(['2024-12-31','2025-01-01','2025-01-07','2025-01-08']),'sensor_value':[1000,10,20,1000]})
    d=sensor_features(rows,sensor,{'EI__TI':False})
    assert d.iloc[0]['EI__TI__7d_mean']==15
    assert d.iloc[0]['EI__TI__1d_mean']==20
    sensor.variable_id='CR__X';sensor.sensor_value=[1,0,1,999]
    d=sensor_features(rows,sensor,{'CR__X':True})
    assert d.iloc[0]['CR__X__7d_active_fraction']==.5


def test_facility_partition_deterministic():
    d=pd.DataFrame({'facility_id':list('abcdefghi')*3})
    a=facility_split(d);assert a==facility_split(d)
    assert set(a['train']).isdisjoint(a['test'])
    assert set().union(*map(set,a.values()))==set(d.facility_id)
    with pytest.raises(ValueError):facility_split(d.iloc[:4])


def test_search_preserves_base_and_deduplicates():
    assert initial_groups('a',['a','b'])==[('a',),('a','b'),('b',)]
    assert neighbours(('a','b'),['a','b','c'],'a')=={('a',),('a','b','c')}


def test_catalog_foreign_keys_and_unverified_semantics():
    source,v,c,m=build_catalog()
    assert not v.Variable_ID.duplicated().any()
    assert set(m.Variable_ID)==set(v.Variable_ID)
    assert set(m.Category_ID)<=set(c.Category_ID)
    assert v[v.Variable_ID.str.startswith('EO__WD')].Availability_Status.eq('needs_review').all()
    assert not c.AX_Category.eq('Constraint').any() # No invented source columns.


def test_xlsx_strings_not_formulas_and_valid_xml(tmp_path):
    p=tmp_path/'r.xlsx';write_xlsx(p,{'Results':pd.DataFrame({'x':['=1+1','한글'], 'v':[1.2,None]})})
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():ET.fromstring(z.read(n))
        data=z.read('xl/worksheets/sheet1.xml').decode()
        assert '<f>' not in data and '=1+1' in data


def test_tft_only_registered_features_and_no_eval_target_dependence():
    pytest.importorskip('pytorch_forecasting')
    from AXData.ax_catalog_experiments.tft import TFTAdapter
    d=pd.DataFrame({'facility_id':['f']*16,'crop_sn':['c']*16,'sample_num':['1']*16,'feature_date':pd.date_range('2025-01-01',periods=16),'target_date':pd.date_range('2025-01-02',periods=16),'x':np.arange(16,dtype=float)+1,'target':np.arange(16,dtype=float)+2})
    params={'hidden_size':4,'attention_head_size':1,'hidden_continuous_size':4,'dropout':0.,'learning_rate':.001,'batch_size':8,'max_epochs':1}
    a=TFTAdapter(['x'],params,42,['facility_id','crop_sn','sample_num'],'x')
    a.fit(d.iloc[:12],'target',fixed_epochs=1)
    assert a.dataset.reals==['x']
    pred=a.predict(d.iloc[12:])
    changed=d.iloc[12:].copy();changed['target']=9999
    assert np.allclose(pred,a.predict(changed))


def test_engine_resume_uses_same_rows_and_does_not_open_test(tmp_path):
    from AXData.ax_catalog_experiments.engine import ExperimentEngine
    prepared=tmp_path/'prepared';prepared.mkdir()
    dates=pd.date_range('2025-01-01',periods=12)
    train=pd.DataFrame({'facility_id':['train']*12,'row_id':[f'a{i}' for i in range(12)],'feature_date':dates,'target_date':dates+pd.Timedelta(days=1),'current':np.arange(12)+1.,'target':np.arange(12)+2.})
    val=train.copy();val.facility_id='val';val.row_id=[f'b{i}' for i in range(12)]
    train.to_parquet(prepared/'train.parquet');val.to_parquet(prepared/'validation.parquet')
    (prepared/'test.parquet').write_text('INTENTIONALLY UNREADABLE BEFORE FREEZE')
    spec={'target_id':'test','current':'current','categories':{'base':['current']},'dataset_id':'test','entity':['facility_id']}
    (prepared/'manifest.json').write_text(json.dumps(spec))
    engine=ExperimentEngine(prepared,tmp_path/'run','smoke');engine.run()
    result=list((tmp_path/'run/runs').glob('*/result.json'))
    assert len(result)==1
    r=json.loads(result[0].read_text());assert r['Status']=='success'
    assert r['N_Eval']==len(val) and r['Features']==['current']
    mtime=result[0].stat().st_mtime_ns
    ExperimentEngine(prepared,tmp_path/'run','smoke').run()
    assert result[0].stat().st_mtime_ns==mtime
    pred=pd.read_parquet(result[0].parent/'predictions.parquet')
    from common_regression.metrics import regression_metrics
    assert regression_metrics(pred.target,pred.prediction)==r['Metrics']
