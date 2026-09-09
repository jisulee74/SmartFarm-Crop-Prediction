from __future__ import annotations
import argparse
import json
from pathlib import Path
from .catalog import build_catalog,snapshot_db
from .data import prepare_all
from .report import build_report


def main():
    p=argparse.ArgumentParser(description='AX catalog / reproducible category experiments')
    p.add_argument('command',choices=['catalog','snapshot-db','prepare','run','report','build','run-all'])
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs')
    p.add_argument('--target',default='tomato_total_flower')
    p.add_argument('--profile',choices=['smoke','full'],default='full')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='snapshot-db':
        result=snapshot_db(a.output/'source_snapshot');(a.output/'db_status.json').write_text(json.dumps(result,indent=2));print(result)
        return
    if a.command in {'prepare','build'}:
        prepared=a.output/'prepared';prepared.mkdir(exist_ok=True)
        print(json.dumps(prepare_all(prepared,build_catalog()[1]),ensure_ascii=False,indent=2),flush=True)
    if a.command=='run-all':
        from .engine import ExperimentEngine
        from .data import TARGETS
        status=[]
        for target in TARGETS:
            prepared=a.output/'prepared'/target
            record={'Target_ID':target,'Status':'running'}
            status.append(record)
            (a.output/'campaign_status.json').write_text(json.dumps(status,indent=2))
            try:
                ExperimentEngine(prepared,a.output/'experiments'/target/a.profile,a.profile).run()
                record['Status']=json.loads((a.output/'experiments'/target/a.profile/'status.json').read_text())['status']
            except Exception as e:
                record.update(Status='failed',Reason=f'{type(e).__name__}: {e}')
            (a.output/'campaign_status.json').write_text(json.dumps(status,indent=2))
            build_report(a.output)
        return
    if a.command=='run':
        from .engine import ExperimentEngine
        prepared=a.output/'prepared'/a.target
        if not (prepared/'manifest.json').exists():raise SystemExit(f'Target not prepared: {a.target}; see prepared/target_status.csv')
        ExperimentEngine(prepared,a.output/'experiments'/a.target/a.profile,a.profile).run()
    if a.command in {'catalog','report','build','run'}:print(json.dumps(build_report(a.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
