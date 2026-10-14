import json
from pathlib import Path

import numpy as np
import pandas as pd

from sync_screening_analysis import stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'ax_catalog_experiments/outputs/exhaustive_e6_semantic_screening_v1'


def bundle():return json.loads((HERE/'ax_direction/dashboard_bundle.json').read_text())


def test_provenance_and_no_validation_error_fabrication():
    b=bundle();t=b['tables'];assert b['metadata']['passed']
    assert b['metadata']['source_campaign']==ROOT.name
    assert len(t['baseline_performance'])==16 and len(t['selected_models'])==8
    assert len(t['model_comparison'])==192
    assert {r['split'] for r in t['error_strata']}=={'test'}
    assert {r['split'] for r in t['crossed_error_strata']}=={'test'}
    for name in ['baseline_performance','error_strata','crossed_error_strata','model_comparison','category_effect_summary']:
        assert all(r['source']==ROOT.name for r in t[name])
    for r in t['target_diagnosis']:
        if r['split']=='validation':assert '저장하지 않아' in r['finding']


def test_partitions_count_unique_rows_not_seed_repetitions():
    t=bundle()['tables'];df=pd.DataFrame(t['error_strata'])
    for (target,dimension),g in df.groupby(['target','dimension']):
        full=next(r for r in t['baseline_performance'] if r['target']==target and r['split']=='test')
        assert g.n.sum()==full['n']
        assert np.isclose(g.row_share.sum(),1)
        assert np.isclose(g.squared_error_share.sum(),1)
    for (target,split),g in pd.DataFrame(t['target_distribution']).groupby(['target_id','split']):
        assert g.n.sum()==g.total_n.iloc[0]
        assert np.isclose(g.rate.sum(),1)


def test_current_missing_baseline_and_change_kpi():
    b=bundle();t=b['tables']
    for target in {r['target'] for r in t['baseline_performance']}:
        spec=json.loads((ROOT/'prepared'/target/'manifest.json').read_text())
        cols=[f'count_{i}__current' for i in spec['components']]
        train=pd.read_parquet(ROOT/'prepared'/target/'train.parquet',columns=cols)
        fallback=train.sum(axis=1,min_count=len(cols)).median()
        for split in ['validation','test']:
            d=pd.read_parquet(ROOT/'prepared'/target/f'{split}.parquet',columns=['target',*cols])
            cur=d[cols].sum(axis=1,min_count=len(cols))
            base=next(r for r in t['baseline_performance'] if r['target']==target and r['split']==split)
            assert base['n']==len(d)
            assert base['persistence_missing_n']==int(cur.isna().sum())
            assert base['persistence_fallback_value']==fallback
            assert np.isclose(base['persistence_rmse'],stats(d.target,cur.fillna(fallback))['rmse'])
            assert np.isclose(base['zero_r2'],stats(d.target,np.zeros(len(d)))['r2'])
            if split=='test':
                changed=(cur.notna()&d.target.ne(cur)).sum()
                row=next(r for r in t['error_strata'] if r['target']==target and r['dimension']=='changed_binary' and r['level']=='changed')
                assert row['n']==changed
                kpi=next(r for r in t['kpi_planning_reference'] if r['target']==target)
                assert kpi['reference_change_n']==changed
                assert kpi['reference_change_mae']==row['mae_mean']
                assert np.isclose(kpi['illustrative_15pct_change_mae'],row['mae_mean']*.85)


def test_overall_model_is_selected_only_by_validation():
    rank=pd.read_csv(ROOT/'validation_rankings.csv');t=bundle()['tables']
    for row in t['selected_models']:
        best=rank[rank.Target_ID.eq(row['target'])].sort_values(['Validation_RMSE','Validation_MAE','Feature_Count','Combination_ID','Model_Type']).iloc[0]
        assert row['model']==best.Model_Type and row['group_id']==best.Combination_ID
        for split in ['validation','test']:
            base=next(r for r in t['baseline_performance'] if r['target']==row['target'] and r['split']==split)
            source=next(r for r in t['model_comparison'] if r['target']==row['target'] and r['split']==split and r['variant']=='bounded' and r['model']==row['model'])
            for metric in ['rmse','mae','r2','ccc']:assert np.isclose(base[metric+'_mean'],source[metric+'_mean'])


def test_category_pairs_differ_by_one_group():
    p=pd.read_csv(HERE/'ax_direction/screening_analysis/paired_category_comparisons.csv')
    cat=json.loads((HERE/'cache/combination_catalog.json').read_text())
    groups={c['combination_id']:set(c['groups']) for c in cat['combinations']}
    assert len(p)==8*6*7*63
    for row in p.itertuples():
        a,b=groups[row.base_combination],groups[row.expanded_combination]
        assert a<b and b-a=={row.added_group}
