"""Durable local coordinator for concurrent training and ordered publication."""
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from .prepare import OUT, prepare
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json


def lane(mode):
    root=OUT/mode;root.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for stage in ['run','report','publish']:
        state=root/'pipeline.json'
        atomic_json(state,{'mode':mode,'stage':stage,'status':'running','updated':time.time()})
        with (root/'pipeline.log').open('a') as log:
            for attempt in range(1,7 if stage=='publish' else 2):
                process=subprocess.Popen([sys.executable,'-u','-m',
                    f'AXData.ax_catalog_experiments.upper_truss_full.{stage}',mode],
                    stdout=log,stderr=subprocess.STDOUT,env=env)
                atomic_json(state,{'mode':mode,'stage':stage,'status':'running','pid':process.pid,
                                  'attempt':attempt,'updated':time.time()})
                code=process.wait()
                if code==0:break
                if stage=='publish' and attempt<6:time.sleep(60)
            if code:
                atomic_json(state,{'mode':mode,'stage':stage,'status':'failed','exit_code':code,
                                   'log':str(root/'pipeline.log'),'updated':time.time()})
                return
    atomic_json(root/'pipeline.json',{'mode':mode,'status':'complete','updated':time.time()})


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        assert json.loads((OUT/'setup_verification.json').read_text())['status']=='passed'
        atomic_json(OUT/'supervisor.json',{'pid':os.getpid(),'status':'running','started':time.time(),
                    'modes':['quick','formal'],'publication_authorized':True})
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lane,['quick','formal']))
        states={m:json.loads((OUT/m/'pipeline.json').read_text()) for m in ['quick','formal']}
        atomic_json(OUT/'supervisor.json',{'pid':os.getpid(),'status':'finished','pipelines':states})
