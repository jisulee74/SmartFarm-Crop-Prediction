import json
from pathlib import Path
import pytest
from AXData.ax_catalog_experiments.stability import parallel,engine,report
from AXData.ax_catalog_experiments.stability.policy import POLICY
from AXData.ax_catalog_experiments.core import digest

def test_worker_keeps_heartbeat_target_local_and_suppresses_report(tmp_path,monkeypatch):
    original_atomic=engine.atomic
    original_report=report.report
    class Fake:
        def __init__(self,root,target):self.root=root;self.target=target
        def run(self):
            engine.atomic(self.root/'heartbeat.json',{'Status':'running'})
            engine.atomic(self.root/'experiments'/self.target/'status.json',{'status':'complete'})
            report.report(self.root)
        def evaluate(self):return 'ok'
    monkeypatch.setattr(engine,'Campaign',Fake)
    try:
        assert parallel.work(tmp_path,'tomato_first')==0
        assert not (tmp_path/'heartbeat.json').exists()
        assert json.loads((tmp_path/'parallel/workers/tomato_first.json').read_text())['Status']=='complete'
        assert not (tmp_path/'AX_Experiment_Report.xlsx').exists()
    finally:
        engine.atomic=original_atomic;report.report=original_report

def test_stop_at_evaluation_boundary(tmp_path,monkeypatch):
    original_atomic=engine.atomic;original_report=report.report
    (tmp_path/'parallel').mkdir();(tmp_path/'parallel/STOP').touch()
    class Fake:
        def __init__(self,*args):pass
        def run(self):self.evaluate()
        def evaluate(self):raise AssertionError('Must not begin another fit after stop')
    monkeypatch.setattr(engine,'Campaign',Fake)
    try:
        assert parallel.work(tmp_path,'tomato_first')==0
        assert json.loads((tmp_path/'parallel/targets/tomato_first.json').read_text())['status']=='paused'
    finally:
        engine.atomic=original_atomic;report.report=original_report

def test_configuration_rejects_changed_engine(tmp_path,monkeypatch):
    (tmp_path/'diagnostics').mkdir();(tmp_path/'diagnostics/gate.json').write_text(json.dumps({'Status':'passed','Policy_Hash':digest(POLICY)}))
    monkeypatch.setattr(parallel,'TARGETS',{})
    expected=dict(models=engine.MODELS,seeds=engine.SEEDS,config=engine.config(),tft_grid=engine.TFT_GRID,
      implementation_hash=engine.implementation_hash(),effective_poisson_grid=engine.grid('poisson'),prepared_manifest_hashes={})
    (tmp_path/'policy.json').write_text(json.dumps(POLICY))
    (tmp_path/'execution_config.json').write_text(json.dumps(expected))
    assert parallel.check_configuration(tmp_path)==expected
    expected['implementation_hash']='changed'
    (tmp_path/'execution_config.json').write_text(json.dumps(expected))
    with pytest.raises(ValueError,match='changed'):parallel.check_configuration(tmp_path)

def test_scheduler_limits_processes_and_isolates_sessions(tmp_path,monkeypatch):
    from AXData.ax_catalog_experiments.stability import validation
    monkeypatch.setattr(parallel,'check_configuration',lambda r:{'implementation_hash':'same'})
    monkeypatch.setattr(validation,'verify',lambda r:None)
    monkeypatch.setattr(report,'report',lambda r:None)
    monkeypatch.setattr(parallel,'TARGETS',dict.fromkeys(['a','b','c']))
    monkeypatch.setattr(parallel.time,'sleep',lambda _:None)
    active=[];peak=[]
    class Child:
        def __init__(self,args,**kwargs):
            assert kwargs['start_new_session'] is True
            self.target=args[-1];self.pid=100+len(peak);self.polls=0
            active.append(self.target);peak.append(len(active))
            engine.atomic(tmp_path/'parallel/targets'/f'{self.target}.json',{'Target_ID':self.target,'status':'complete'})
        def poll(self):
            self.polls+=1
            if self.polls==1:return None
            if self.target in active:active.remove(self.target)
            return 0
        def wait(self):return 0
    monkeypatch.setattr(parallel.subprocess,'Popen',Child)
    parallel.run(tmp_path,2)
    s=json.loads((tmp_path/'parallel/scheduler.json').read_text())
    assert max(peak)==2 and s['Status']=='finished' and len(s['Finished'])==3
