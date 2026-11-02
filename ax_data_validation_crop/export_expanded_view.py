"""Read saved, verified results into anonymous display assets. Never train or alter results.

Usage: python export_expanded_view.py --campaign PATH --summary PATH --output PATH
The display manifest is bound to the exact published summary hash; mismatches fail closed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ['rmse', 'mae', 'r2', 'ccc']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))

def aggregate(rows):
    answer = {}
    for key in METRICS:
        vals = [r[key] for r in rows if r[key] is not None]
        answer[key + '_mean'] = float(np.mean(vals)) if vals else None
        answer[key + '_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return answer

def metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    mse = float(np.mean((y - p) ** 2))
    var = float(np.var(y)); denom = var + float(np.var(p)) + float((y.mean() - p.mean()) ** 2)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(y-p))),
                r2=1-mse/var if var > 0 else None,
                ccc=float(2*np.mean((y-y.mean())*(p-p.mean()))/denom) if denom > 0 else None)

def export(campaign, summary_path, output):
    assert not output.exists(), 'Use a new output directory; do not overwrite a display release'
    bundle = json.loads(summary_path.read_text()); mode = bundle['mode']; source = campaign / mode
    verification = json.loads((source / 'verification.json').read_text())
    assert verification['status'] == 'passed' and verification['summary_hash'] == sha(summary_path)
    assert sha(source / 'summary.json') == sha(summary_path)
    assert json.loads((source / 'progress.json').read_text())['phase'] == 'complete'
    frozen = json.loads((source / 'frozen_winners.json').read_text())
    manifest = json.loads((campaign / 'manifest.json').read_text())
    source_hashes = {}; counts = {'validation_combinations': 0, 'prediction_files': 0, 'prediction_rows': 0, 'checked_truss_metrics': 0}
    def read(path):
        source_hashes[str(path)] = sha(path)
        return json.loads(path.read_text())
    matrices = {}; fac = {}; cycles = {}; samples = {}
    for crop in bundle['crops']:
        for split in ['validation', 'test']:
            path = campaign / crop / (split + '.parquet')
            assert sha(path) == manifest['crops'][crop]['hashes'][split]
            frame = pd.read_parquet(path)
            matrices[crop, split] = frame
            for r in frame.itertuples():
                f = str(r.facility_id); c = (f, str(r.crop_sn)); s = (*c, str(r.sample_num))
                fac.setdefault(f, f'F{len(fac)+1:02d}')
                cycles.setdefault(c, f'C{len(cycles)+1:02d}')
                if s not in samples:
                    samples[s] = str(1 + sum(k[:2] == c for k in samples))
    columns = ['facility_id', 'crop_sn', 'sample_num', 'row_id', 'truss', 'feature_date', 'target_date', 'current', 'target', 'prediction_raw', 'prediction_bounded']
    def prediction_file(record, crop, split, model):
        path = source / 'runs' / record['run_id'] / 'predictions.parquet'
        source_hashes[str(path)] = sha(path)
        data = pd.read_parquet(path); truth = matrices[crop, split]
        for col in ['row_id', 'facility_id', 'crop_sn', 'sample_num', 'truss', 'feature_date', 'target_date', 'current', 'target']:
            pd.testing.assert_series_equal(data[col], truth[col])
        assert np.isfinite(data[['prediction', 'prediction_raw']]).all().all()
        np.testing.assert_array_equal(data.prediction, np.maximum(data.prediction_raw, 0))
        raw_trusses = {}
        for truss, group in data.groupby('truss'):
            check = metrics(group.target, group.prediction)
            stored = next(row for row in record['per_truss'] if row['truss'] == truss)
            for key in METRICS:
                if check[key] is None: assert stored[key] is None
                else: np.testing.assert_allclose(check[key], stored[key], rtol=1e-10, atol=1e-10)
            raw_trusses[str(truss)] = metrics(group.target, group.prediction_raw)
            counts['checked_truss_metrics'] += 4
        rows = []
        for i, r in enumerate(data.itertuples()):
            f = str(r.facility_id); c = (f, str(r.crop_sn)); s = (*c, str(r.sample_num))
            rows.append([fac[f], cycles[c], samples[s], f'{crop}_{split}_{i}', int(r.truss),
                         str(r.feature_date)[:10], str(r.target_date)[:10], float(r.current), float(r.target), float(r.prediction_raw), float(r.prediction)])
        name = f'predictions/{crop}_{model}_{split}_{record["seed"]}.json'
        write(output / name, dict(columns=columns, rows=rows, crop=crop, model=model, split=split, seed=record['seed'], combo=record['combo']))
        counts['prediction_files'] += 1; counts['prediction_rows'] += len(rows)
        return name, raw_trusses
    index = dict(schema=1, summary_sha256=sha(summary_path), campaign=bundle['campaign'], mode=mode,
                 seeds=bundle['seeds'], limits=bundle['limits'], models={}, groups={}, selected=frozen['overall'])
    for crop, crop_spec in bundle['crops'].items():
        index['models'][crop] = {}; index['groups'][crop] = manifest['crops'][crop]['groups']
        for model_spec in crop_spec['models']:
            model = model_spec['model']; winner = next(w for w in frozen['winners'] if w['crop'] == crop and w['model'] == model)
            records = {}
            for path in sorted((source / 'summaries' / crop / model).glob('*.json')):
                s = read(path); assert s['status'] == 'success'
                runs = [read(source / 'runs' / key / 'result.json') for key in s['run_ids']]
                assert {r['seed'] for r in runs} == set(bundle['seeds'])
                assert all(r['stage'] == 'validation' and r['status'] == 'success' and r['selected'] == runs[0]['selected'] for r in runs)
                expected = next(row for row in bundle['comparison'] if row['crop'] == crop and row['model'] == model and row['groups'] == '+'.join(s['groups']))
                for k, v in aggregate([r['metrics'] for r in runs]).items():
                    metric, stat = k.rsplit('_', 1)
                    np.testing.assert_allclose(v, expected['metrics'][metric][stat], rtol=1e-10, atol=1e-10) if v is not None else None
                trusses = {}
                for t in range(1, crop_spec['limit'] + 1):
                    part = [next(row for row in r['per_truss'] if row['truss'] == t) for r in runs]
                    assert len({r['n'] for r in part}) == 1
                    trusses[str(t)] = dict(bounded=aggregate(part), n_eval=part[0]['n'], positive_n=part[0]['positive_n'])
                item = dict(groups=s['groups'], features=s['features'], selected=runs[0]['selected'], params=s['params'],
                            best_epoch=s['epochs'], validation=dict(trusses=trusses, predictions={}), test=None)
                if s['combo'] == winner['combo']:
                    raw = []
                    for run in runs:
                        name, raw_by_truss = prediction_file(run, crop, 'validation', model)
                        item['validation']['predictions'][str(run['seed'])] = dict(path=name, sha256=sha(output / name)); raw.append(raw_by_truss)
                    for t in trusses: trusses[t]['raw'] = aggregate([r[t] for r in raw])
                    tests = []
                    for seed in bundle['seeds']:
                        identity = dict(crop=crop, combo=winner['combo'], model=model, params=winner['params'], seed=seed,
                                        stage='test', fold=None, epochs=winner['epochs'])
                        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:24]
                        record = read(source / 'runs' / key / 'result.json')
                        assert record['status'] == 'success' and record['combo'] == winner['combo']
                        tests.append(record)
                    test_trusses = {}
                    for t in range(1, crop_spec['limit']+1):
                        part = [next(row for row in r['per_truss'] if row['truss'] == t) for r in tests]
                        test_trusses[str(t)] = dict(bounded=aggregate(part), n_eval=part[0]['n'], positive_n=part[0]['positive_n'])
                        public = next(row for row in model_spec['per_truss'] if row['truss'] == t)
                        for key in METRICS:
                            if public['metrics'][key]['mean'] is not None:
                                np.testing.assert_allclose(test_trusses[str(t)]['bounded'][key+'_mean'], public['metrics'][key]['mean'], rtol=1e-10, atol=1e-10)
                    item['test'] = dict(trusses=test_trusses, predictions={}); raw = []
                    for run in tests:
                        name, raw_by_truss = prediction_file(run, crop, 'test', model)
                        item['test']['predictions'][str(run['seed'])] = dict(path=name, sha256=sha(output / name)); raw.append(raw_by_truss)
                    for t in test_trusses: test_trusses[t]['raw'] = aggregate([r[t] for r in raw])
                records[s['combo']] = item; counts['validation_combinations'] += 1
            assert len(records) == 127
            name = f'{crop}_{model}.json'
            write(output / name, dict(winner=winner['combo'], combinations=records))
            index['models'][crop][model] = dict(path=name, sha256=sha(output / name), winner=winner['combo'])
    assert counts['validation_combinations'] == 1524
    for path, h in source_hashes.items(): assert sha(Path(path)) == h, path
    write(output / 'manifest.json', index)
    audit = dict(status='passed', counts=counts, source_files_unchanged=len(source_hashes), summary_sha256=index['summary_sha256'])
    print(json.dumps(audit))
    return audit, source_hashes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    audit, hashes = export(args.campaign, args.summary, args.output)
    write(args.audit, audit)
    write(args.audit.with_name('source_hashes.json'), hashes)
