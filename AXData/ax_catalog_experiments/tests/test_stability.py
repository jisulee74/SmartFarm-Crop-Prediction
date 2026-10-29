import json
import numpy as np
import pandas as pd
import pytest
from AXData.ax_catalog_experiments.stability.policy import StablePreprocessor,constrain,score,upper_bound
from AXData.ax_catalog_experiments.stability.data import PARTS,ENTITY,legacy_feature
from AXData.ax_catalog_experiments.stability.adapters import adapter

def test_near_constant_shift_keeps_width_and_caps_without_refit():
    train=pd.DataFrame({'a':[1.,1.+1e-12,1.], 'b':[None,4.,6.], 'c':[0.,0.,0.]})
    prep=StablePreprocessor(list(train)).fit(train)
    before=prep.mean.copy()
    val=pd.DataFrame({'a':[1e9], 'b':[None], 'c':[900.]})
    result=prep.transform(val)
    assert result.shape==(1,3) and np.isfinite(result).all()
    assert result[0,0]==10 and result[0,2]==10
    pd.testing.assert_series_equal(before,prep.mean)
    assert prep.diagnostics(val).Input_Clip_Rate.tolist()==[1,0,1]

def test_empty_feature_fails_no_silent_removal():
    with pytest.raises(ValueError,match='entirely missing'):
        StablePreprocessor(['x']).fit(pd.DataFrame({'x':[None,np.inf]}))

def test_output_guard_and_train_component_upper():
    train=pd.DataFrame({'target_truss_1':[2,4],'target_truss_2':[6,3],'target_truss_3':[1,2]})
    assert upper_bound(train,[1,2,3])==12
    out,unstable=constrain([-1,5,202e12],12)
    assert out.tolist()==[0,5,12] and unstable
    with pytest.raises(FloatingPointError):constrain([np.inf],12)
    assert score([0,0],[0,1])['r2'] is None

def test_legacy_mapping():
    assert legacy_feature('FlowerA_legacy::current_total_flower_count','flower_count')=='flower_count__current'
    assert legacy_feature('FruitA_legacy::fruit_count_roll4_std','fruit_count')=='fruit_count__roll4_std'

@pytest.mark.parametrize('model',['poisson','random_forest','catboost','mlp','tabm'])
def test_stable_adapter_extreme_input(model):
    import torch
    torch.set_num_threads(2)
    d=pd.DataFrame({'x':np.arange(16,dtype=float),'flat':np.ones(16)+np.arange(16)*1e-14,'target':np.arange(16)%4+1.})
    params={'poisson':{'alpha':1.,'max_iter':2000},'random_forest':{'n_estimators':3,'n_jobs':1},'catboost':{'iterations':3,'thread_count':1},'mlp':{'max_epochs':2,'hidden':[8]},'tabm':{'max_epochs':2,'k':2,'n_blocks':1}}
    a=adapter(model,['x','flat'],params[model],42,ENTITY,'x');a.fit(d,'target',validation=d)
    val=d.copy();val['flat']=1e10
    pred=a.predict(val)
    assert np.isfinite(pred).all() and len(pred)==len(val)
    assert a.preprocessor.transform(val).shape==(16,2)

def test_tft_registered_inputs_and_target_isolation():
    import torch
    torch.set_num_threads(2)
    dates=pd.date_range('2024-01-01',periods=8)
    d=pd.DataFrame({'facility_id':['a']*8,'crop_sn':['c']*8,'sample_num':['1']*8,'feature_date':dates,'target_date':dates+pd.Timedelta(days=1),'x':np.arange(8,dtype=float),'target':np.arange(8)%3+1.})
    params=dict(hidden_size=4,attention_head_size=1,hidden_continuous_size=4,dropout=0.,learning_rate=.001,batch_size=8,max_epochs=1)
    a=adapter('tft',['x'],params,42,ENTITY,'not_an_input');a.fit(d,'target',validation=d.assign(facility_id='b'))
    v=d.assign(facility_id='b');p=a.predict(v);v.target=1e12;q=a.predict(v)
    assert np.isfinite(p).all() and np.allclose(p,q)
    assert a.dataset.reals==['x']

def test_poisson_floor_and_dedup():
    from AXData.ax_catalog_experiments.stability.engine import grid
    a=adapter('poisson',['x'],{'alpha':0,'max_iter':2000},42,ENTITY,'x')
    assert a.params['alpha']==.1
    assert [p['alpha'] for p in grid('poisson')]==[.1,1.]

def test_sum_upper_includes_current_observations():
    d=pd.DataFrame({'target_truss_1':[1,2], 'count_1':[7,1]})
    assert upper_bound(d,[1])==7

def test_gate_prevents_campaign_without_pass(tmp_path):
    from AXData.ax_catalog_experiments.stability.engine import run_all
    (tmp_path/'diagnostics').mkdir()
    (tmp_path/'diagnostics/gate.json').write_text(json.dumps({'Status':'blocked'}))
    with pytest.raises(ValueError,match='gate'):run_all(tmp_path)


def test_engine_preserves_rows_raw_metrics_and_cache(tmp_path):
    from AXData.ax_catalog_experiments.stability.engine import Campaign
    from AXData.ax_catalog_experiments.core import file_hash
    target='tomato_first';p=tmp_path/'prepared'/target;p.mkdir(parents=True)
    d=pd.DataFrame({'facility_id':['train']*9,'crop_sn':['c']*9,'sample_num':['1']*9,
      'feature_date':pd.date_range('2024-01-01',periods=9),'target_date':pd.date_range('2024-01-02',periods=9),
      'row_id':[str(i) for i in range(9)],'x':np.arange(9,dtype=float),'target':np.arange(9)%3+1.,'target_truss_1':np.arange(9)%3+1.})
    for split in ['train','validation','test']:
        f=d.assign(facility_id=split,row_id=[split+str(i) for i in range(9)])
        f.to_parquet(p/f'{split}.parquet',index=False)
    spec=dict(target_id=target,entity=ENTITY,current='x',components=[1],groups={'g':{'features':['x'],'categories':{'c':['x']}}},dataset_hashes={s:file_hash(p/f'{s}.parquet') for s in ['train','validation','test']})
    (p/'manifest.json').write_text(json.dumps(spec))
    c=Campaign(tmp_path,target)
    r=c.evaluate('g','random_forest',{'n_estimators':3},42,'validation')
    assert r['Status']=='success' and r['N_Eval']==9
    pred=pd.read_parquet(c.out/'runs'/r['Run_ID']/'predictions.parquet')
    assert set(pred.row_id)==set(c.val.row_id)
    assert score(pred.target,pred.prediction_raw)==r['Metrics']['raw']
    assert score(pred.target,pred.prediction_bounded)==r['Metrics']['bounded']
    assert c.evaluate('g','random_forest',{'n_estimators':3},42,'validation')==r
