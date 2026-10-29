"""Process-isolated target scheduling; scientific engine and cache identities unchanged."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import engine
from .data import SOURCE,TARGETS
from .policy import POLICY
from ..core import digest,freeze,file_hash

class PauseRequested(BaseException): pass

def check_configuration(root):
    gate=json.loads((root/'diagnostics/gate.json').read_text())
    if gate.get('Status')!='passed' or gate.get('Policy_Hash')!=digest(POLICY):
        raise ValueError('Current diagnostic gate must pass')
    expected=dict(models=engine.MODELS,seeds=engine.SEEDS,config=engine.config(),tft_grid=engine.TFT_GRID,
      implementation_hash=engine.implementation_hash(),effective_poisson_grid=engine.grid('poisson'),
      prepared_manifest_hashes={t:file_hash(root/'prepared'/t/'manifest.json') for t in TARGETS})
    existing=json.loads((root/'execution_config.json').read_text())
    if existing!=expected:raise ValueError('Scientific configuration/code/data changed; cannot reuse campaign')
    if json.loads((root/'policy.json').read_text())!=POLICY:raise ValueError('Policy changed')
    return expected

def work(root,target):
    import contextlib
    from . import report as reporting
    # A target is exclusively owned by one subprocess; reporting has a single coordinator writer.
    reporting.report=lambda *args,**kwargs:None
    original_atomic=engine.atomic
    state=root/'parallel/workers'/f'{target}.json'
    def atomic(path,value):
        if Path(path)==root/'heartbeat.json':
            path=state
        original_atomic(path,value)
    engine.atomic=atomic
    class Worker(engine.Campaign):
        def evaluate(self,*args,**kwargs):
            if (root/'parallel/STOP').exists():raise PauseRequested()
            return super().evaluate(*args,**kwargs)
    lock_path=root/'experiments'/target/'worker.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            Worker(root,target).run()
            result=json.loads((root/'experiments'/target/'status.json').read_text())
        except PauseRequested:
            result=dict(status='paused',reason='Stop requested at persisted-run boundary')
        except Exception as e:result=dict(status='blocked',reason=f'{type(e).__name__}: {e}')
        original_atomic(state,dict(Target_ID=target,PID=os.getpid(),Status=result['status'],Completed_At=time.time()))
        original_atomic(root/'parallel/targets'/f'{target}.json',dict(Target_ID=target,**result))
    return 0 if result['status'] in ['complete','complete_with_exclusions','paused'] else 1

def status(root):
    from .report import progress
    result=progress(root)
    result.pop('Heartbeat',None)
    result['Execution_Mode']='parallel_targets'
    result['Workers']=[json.loads(p.read_text()) for p in sorted((root/'parallel/workers').glob('*.json'))]
    result['Scheduler']=json.loads((root/'parallel/scheduler.json').read_text()) if (root/'parallel/scheduler.json').exists() else None
    # The historical serial mean is not a valid parallel wall-clock ETA.
    result['Estimated_Remaining_Seconds']=None
    result['ETA_Note']='Parallel wall-clock ETA not yet calibrated; per-model costs differ'
    result['Progress_Percent']=round(100*result['Completed_Runs']/result['Planned_Run_Upper_Bound'],2)
    return result

def run(root,workers):
    from .validation import verify
    from .report import report
    root.mkdir(parents=True,exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=check_configuration(root);verify(root)
        directory=root/'parallel';directory.mkdir(exist_ok=True)
        old=directory/'scheduler.json'
        if old.exists():
            for pid in json.loads(old.read_text()).get('Active',{}).values():
                proc=Path(f'/proc/{pid}/cmdline')
                if proc.exists() and b'stability.parallel' in proc.read_bytes():
                    raise RuntimeError('Previous workers remain active; request stop and wait before resuming')
        stop=directory/'STOP';stop.unlink(missing_ok=True)
        def request_stop(*_):stop.touch()
        previous={sig:signal.signal(sig,request_stop) for sig in [signal.SIGINT,signal.SIGTERM]}
        engine.atomic(directory/'runtime.json',dict(workers=workers,engine_hash=config['implementation_hash'],scheduler_hash=file_hash(Path(__file__)),started_at=time.time()))
        queue=[]
        for target in TARGETS:
            p=root/'experiments'/target/'status.json'
            if p.exists() and json.loads(p.read_text()).get('status') in ['complete','complete_with_exclusions']:continue
            queue.append(target)
        active={};finished=[]
        try:
            while queue or active:
                while queue and len(active)<workers and not stop.exists():
                    target=queue.pop(0)
                    log_path=directory/'logs'/f'{target}.log';log_path.parent.mkdir(parents=True,exist_ok=True)
                    with log_path.open('a') as log:
                        process=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker','--output',str(root),'--target',target],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[target]=process
                    print('started',target,'pid',process.pid,flush=True)
                changed=False
                for target,process in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    p=directory/'targets'/f'{target}.json'
                    record=json.loads(p.read_text()) if p.exists() else dict(Target_ID=target,status='blocked',reason=f'Worker exited {code} without completion record')
                    finished.append(record);del active[target];changed=True
                    print('finished',target,record['status'],flush=True)
                engine.atomic(directory/'scheduler.json',dict(PID=os.getpid(),Status='stopping' if stop.exists() else 'running',Workers=workers,Active={t:p.pid for t,p in active.items()},Queued=queue,Finished=finished,Updated_At=time.time()))
                if changed:
                    engine.atomic(root/'campaign_status.json',finished)
                    report(root)
                if stop.exists() and not active:break
                time.sleep(2)
            engine.atomic(directory/'scheduler.json',dict(PID=os.getpid(),Status='paused' if stop.exists() else 'finished',Workers=workers,Active={},Queued=queue,Finished=finished,Updated_At=time.time()))
            report(root)
        finally:
            # On coordinator failure, ask surviving workers to finish their current persisted run.
            if active:
                stop.touch()
                for process in active.values():process.wait()
            for sig,handler in previous.items():signal.signal(sig,handler)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','status','stop','check','worker'])
    p.add_argument('--output',type=Path,default=SOURCE/'stability_v2')
    p.add_argument('--workers',type=int,choices=range(1,5),default=2)
    p.add_argument('--target',choices=list(TARGETS))
    a=p.parse_args();root=a.output.resolve()
    if a.command=='check':
        check_configuration(root)
        print('Compatible: engine, policy, seeds, models, grids, dataset manifests unchanged')
    elif a.command=='run':run(root,a.workers)
    elif a.command=='worker':
        if not a.target:p.error('--target required for worker')
        raise SystemExit(work(root,a.target))
    elif a.command=='stop':
        (root/'parallel').mkdir(exist_ok=True);(root/'parallel/STOP').touch()
        print('Stop requested; each worker finishes its current run before pausing')
    else:print(json.dumps(status(root),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
