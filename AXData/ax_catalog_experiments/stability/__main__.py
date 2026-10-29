from pathlib import Path
import argparse,json
from .data import prepare,SOURCE
from .diagnostics import diagnose
from .engine import run_all
from .report import report,progress

def main():
    p=argparse.ArgumentParser(description='Stability v2: 8 targets, every historical group, six models/five seeds')
    p.add_argument('command',choices=['prepare','diagnose','run-all','report','status','verify','all'])
    p.add_argument('--output',type=Path,default=SOURCE/'stability_v2')
    a=p.parse_args()
    if a.command in ['prepare','all']:prepare(a.output)
    if a.command in ['diagnose','all']:
        gate=diagnose(a.output);report(a.output)
        if gate['Status']!='passed':raise SystemExit('Diagnostic gate blocked; inspect diagnostics/gate.json')
    if a.command in ['verify','run-all','all']:
        from .validation import verify
        verify(a.output)
    if a.command in ['run-all','all']:run_all(a.output)
    if a.command in ['report','all']:report(a.output)
    print(json.dumps(progress(a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
