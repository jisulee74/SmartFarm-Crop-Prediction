import pandas as pd
from AXData.ax_catalog_experiments.display_ids import simplify_ids


def fixture(extra=False):
    return {'Experiments':pd.DataFrame([{'Experiment_ID':'hash_e','Target_ID':'tomato_total_flower','Variable_Group_ID':'hash_g'}]+([{'Experiment_ID':'aaa','Target_ID':'strawberry_first','Variable_Group_ID':'aaa_g'}] if extra else [])), 'Results':pd.DataFrame([{'Experiment_ID':'hash_e','Variable_Group_ID':'hash_g','Run_ID':'hash_r'}]),'Category_Comparisons':pd.DataFrame([{'Base_Group_ID':'hash_g','Added_Group_ID':'hash_g','Category_ID':'hash_c'}])}


def test_stable_mapping_and_all_references(tmp_path):
    a=fixture();simplify_ids(a,tmp_path)
    assert a['Experiments'].iloc[0].Experiment_ID=='EXP_001'
    assert a['Results'].iloc[0].Variable_Group_ID=='VG_Flower_001'
    assert a['Category_Comparisons'].iloc[0].Base_Group_ID=='VG_Flower_001'
    b=fixture(True);simplify_ids(b,tmp_path)
    assert b['Experiments'].iloc[0].Experiment_ID=='EXP_001'
    assert b['Experiments'].iloc[1].Experiment_ID=='EXP_002'
    assert b['Experiments'].iloc[1].Variable_Group_ID=='VG_Fruit_001'
    assert b['ID_Mapping'].Original_ID.isin(['hash_e','hash_g','hash_r','hash_c']).sum()==4
