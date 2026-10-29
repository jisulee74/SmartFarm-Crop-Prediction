"""Authorized, serialized publication; formal results cannot be downgraded."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from AXData.ax_catalog_experiments.core import file_hash
from AXData.ax_catalog_experiments.exhaustive.io import atomic_json
from .prepare import OUT, ROOT
from .site import build

REPOSITORY='https://github.com/jisulee74/SmartFarm-Crop-Prediction.git'
PUBLIC='https://jisulee74.github.io/SmartFarm-Crop-Prediction/ax_data_validation_crop/'


def cmd(args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()


def publish(mode):
    root=OUT/mode
    verify=json.loads((root/'verification.json').read_text())
    assert verify['status']=='passed'
    assert file_hash(root/'summary.json')==verify['summary_hash']
    bundle=json.loads((root/'summary.json').read_text())
    assert bundle['mode']==mode and bundle['limits']=={'tomato':20,'strawberry':5}
    site=root/'site';build(bundle,site)
    shutil.copy2(root/'validation_comparison.csv',site/'validation_comparison.csv')
    shutil.copytree(ROOT/'ax_data_validation_crop/src',site/'src',dirs_exist_ok=True)
    (site/'README.md').write_text(
        f"# 화방 확장 실험 결과\n\n캠페인 {OUT.name}, {mode}, {bundle['seed_count']}시드.\n"
        "기본 네 지표는 화방별 지표의 산술평균입니다. 전체 행 지표와 합계 예측 지표는 상세 영역에 구분했습니다.\n"
        "127개 조합은 Validation으로 비교하며, 모델별 선정 조합만 Test에서 평가합니다.\n")
    atomic_json(root/'publication.json',{'status':'ready','mode':mode,'site':str(site)})
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        checkout=OUT/'publication_checkout'
        if not (checkout/'.git').exists():
            cmd(['git','clone','--single-branch','--branch','main',REPOSITORY,str(checkout)])
        assert not cmd(['git','status','--porcelain'],checkout),'Publication checkout has uncommitted changes'
        cmd(['git','pull','--ff-only','origin','main'],checkout)
        destination=checkout/'ax_data_validation_crop'
        remote_path=destination/'summary.json'
        current=json.loads(remote_path.read_text()) if remote_path.exists() else {}
        if current.get('campaign')==OUT.name and current.get('mode')=='formal' and mode=='quick':
            atomic_json(root/'publication.json',{'status':'superseded_by_formal','url':PUBLIC})
            return
        expected={name:file_hash(site/name) for name in ['summary.json','index.html']}
        if not (destination/'summary.json').exists() or file_hash(destination/'summary.json')!=expected['summary.json'] or file_hash(destination/'index.html')!=expected['index.html']:
            for path in site.rglob('*'):
                if path.is_file():
                    target=destination/path.relative_to(site);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,target)
            cmd(['git','add','--','ax_data_validation_crop'],checkout)
            changed=cmd(['git','diff','--cached','--name-only'],checkout).splitlines()
            assert changed and all(p.startswith('ax_data_validation_crop/') for p in changed)
            # Reuse the established deployment identity, without touching its worktree.
            old=ROOT.parent/'repo_deploy/repo'
            for key in ['user.name','user.email']:
                value=cmd(['git','config',key],old)
                cmd(['git','config',key,value],checkout)
            cmd(['git','commit','-m',f'Publish {mode} 20/5-truss exhaustive experiment results'],checkout)
        sha=cmd(['git','rev-parse','HEAD'],checkout)
        cmd(['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential',
             'push','origin','main'],checkout)
        atomic_json(root/'publication.json',{'status':'pushed','commit':sha,'url':PUBLIC})
        deadline=time.time()+1800
        deployed=False
        while time.time()<deadline:
            data=json.loads(cmd(['gh','api',f'repos/jisulee74/SmartFarm-Crop-Prediction/actions/runs?head_sha={sha}']))
            runs=[r for r in data['workflow_runs'] if r['name']=='Deploy GitHub Pages']
            if runs and runs[0]['status']=='completed':
                assert runs[0]['conclusion']=='success',f'Pages workflow failed: {runs[0]["html_url"]}'
                deployed=True;break
            time.sleep(15)
        assert deployed,'Pages deployment timeout'
        for attempt in range(30):
            matches=True
            for name,h in expected.items():
                try:
                    with urllib.request.urlopen(PUBLIC+name+f'?release={sha}&check={attempt}',timeout=30) as response:
                        matches &= hashlib.sha256(response.read()).hexdigest()==h
                except Exception:
                    matches=False
            if matches:break
            time.sleep(15)
        assert matches,'Public files do not match verified artifacts'
        for path in site.rglob('*'):
            if path.is_file():
                target=ROOT/'ax_data_validation_crop'/path.relative_to(site)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        atomic_json(root/'publication.json',{'status':'published_verified','commit':sha,'url':PUBLIC,
                                             'public_hashes':expected,'verified_at':time.time()})
        print('PUBLISHED',mode,PUBLIC,sha,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['quick','formal']);publish(p.parse_args().mode)
