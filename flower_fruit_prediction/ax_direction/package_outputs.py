"""Export web tables; validate the deliverable and make a portable archive."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,zipfile,re,subprocess,shutil,xml.etree.ElementTree as ET,datetime
H=Path(__file__).resolve().parent;T=H/'tables'
checks=[]
def check(name,passed,detail=''):
 checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
 if not passed:raise AssertionError(name+': '+str(detail))
meta=json.loads((H/'audit.json').read_text());check('analysis_passed',meta['passed'])
keep=['baseline_performance','cohort_profile','selected_models','error_strata','crossed_error_strata','target_diagnosis','hypotheses','ax_field_schema','comparison_arms','kpi_planning_reference','category_effect_summary','model_comparison']
bundle={'metadata':meta,'tables':{}}
def records(d):return json.loads(d.to_json(orient='records',force_ascii=False,date_format='iso'))
for name in keep:bundle['tables'][name]=records(pd.read_csv(T/(name+'.csv')))
(H/'dashboard_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':')))
description={
 'baseline_performance':'One frozen overall winner per target and evaluation split; bounded per-seed metrics then mean. Includes baselines/raw sensitivity.',
 'error_strata':'Same frozen winner, per dimension/level; n counts unique evaluation rows, not seed repetitions.',
 'crossed_error_strata':'Test-only descriptive crossing of facility/age/horizon/change; not causal adjustment.',
 'model_seed_metrics':'Recomputed raw and bounded metrics for each model-specific validation winner and persistence; per seed/split.',
 'selected_predictions':'Five seeds of overall validation-selected model predictions with target_id task, target actual count; no averaging performed.',
 'registered_categories':'One row per target/group/category. Category IDs can differ for equivalent meaning; preserve exact feature membership.',
 'paired_category_comparisons':'Validation-only addition of one category within same model and exact category subset; separately tuned; not causal.',
 'kpi_planning_reference':'Reference test metrics and illustrative proposed targets only; not measured To-Be outcomes.',
 'ax_field_schema':'Proposed logical field schema, not collected data; records with missing required scope keys cannot be joined.',
}
column_notes={
 'target':'In CSV metrics tables: task ID. In selected_predictions.parquet: actual numeric count. Use target_id for task in predictions.',
 'target_id':'Prediction task ID in prediction rows.',
 'n':'Unique evaluation observation rows in this stratum; seeds do not multiply n.',
 'rmse_mean':'Arithmetic mean of per-seed RMSE values, not RMSE of the average prediction.',
 'rmse_seed_std':'Sample SD (ddof=1) across seed metrics; not facility/generalization confidence interval.',
 'squared_error_share':'Sum of per-row mean-over-seeds squared error in stratum / same sum in full target/split.',
 'bias_mean':'Average(prediction-actual), then average over seeds; positive=overprediction.',
 'row_share':'Stratum n divided by full target/split n.',
 'posthoc_label_based':'True if segmentation needs future actual target (change/positive); not a forecast-time feature.',
 'level':'Dimension-dependent category; dimensions documented below.',
 'small_n':'n<30 descriptive warning, not a statistically derived universal threshold.',
 'within1_mean':'Fraction of errors within ±1 count, averaged over seeds; diagnostic only.',
 'output_clip_rate':'Fraction of raw != bounded predictions across seeds and rows.',
 'delta_rmse':'Added-category RMSE minus base RMSE; negative means improvement in validation.',
 'skill_vs_persistence_pct':'100*(1-selected_model_rmse_mean/persistence_rmse); negative=worse.',
 'illustrative_10pct_rmse':'0.9*current reference RMSE; planning only; rebaseline on future paired cohort.',
 'record_status':'observed_no_event/event_recorded/unknown must remain distinct.',
}
dims={
 'all':'All rows in selected target/split.', 'facility':'facility_id.', 'crop_cycle':'facility_id / crop_sn composite.',
 'horizon_days':'target_date-feature_date: (0,7],(7,14],(14,28],(28,infinity); retrospective, unknown at original forecast unless schedule exists.',
 'previous_gap_days':'days_since_previous: (0,7],(7,14],(14,28],(28,infinity); missing kept separate.',
 'crop_age_days':'days_since_crop_start <=60,61-120,121-180,181+; descriptive elapsed time, not verified physiological phase.',
 'change':'actual next count-current count: <=-2,(-2,0),0,(0,2),>=2; posthoc.',
 'changed_binary':'actual next count != current count; posthoc.',
 'label_positive':'actual next count >0; posthoc.', 'current_positive':'current count >0, known at feature time.',
 'history_count':'prior/available history_count numeric cut <=1,2-3,4+.',
 'selected_input_missing':'Any nonfinite/missing selected input in evaluation row before imputation.',
 'selected_input_outside_train':'Any selected numeric input outside train min-max; missing counted separately.',
}
dictionary={'version':'2026-09-14','notes':column_notes,'dimensions':dims,'tables':{}}
for p in sorted(T.glob('*.csv')):
 d=pd.read_csv(p)
 dictionary['tables'][p.name]={'rows':len(d),'description':description.get(p.stem,'See the three reports and generating scripts for exact derivation.'),
  'columns':[{'name':k,'dtype':str(v),'description':column_notes.get(k,'')} for k,v in d.dtypes.items()]}
p=pd.read_parquet(T/'selected_predictions.parquet')
dictionary['tables']['selected_predictions.parquet']={'rows':len(p),'description':description['selected_predictions'],'columns':[{'name':k,'dtype':str(v)} for k,v in p.dtypes.items()]}
(H/'data_dictionary.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2,allow_nan=False))
check('web_bundle_roundtrip',len(json.loads((H/'dashboard_bundle.json').read_text())['tables'])==len(keep))
check('predictions_have_all_targets',p.target_id.nunique()==8)
check('predictions_unique_seed_rows',not p.duplicated(['target_id','Eval_Split','row_id','seed']).any())
b=pd.read_csv(T/'baseline_performance.csv'); ss=pd.read_csv(T/'error_strata.csv')
for row in b.itertuples():
 q=p[(p.target_id==row.target)&(p.Eval_Split==row.split)]
 check(row.target+'_'+row.split+'_export_n',q.row_id.nunique()==row.n and q.seed.nunique()==5)
check('eight_test_baselines',len(b[b.split=='test'])==8)
check('expected_tomato_sum',np.isclose(b[(b.target=='tomato_sum123')&(b.split=='test')].rmse_mean.iloc[0],0.89686578,atol=1e-6))
check('expected_strawberry_second',np.isclose(b[(b.target=='strawberry_second')&(b.split=='test')].rmse_mean.iloc[0],1.4944446,atol=1e-6))
# OOXML workbook structure and sheet counts.
with zipfile.ZipFile(H/'분석표_및_AX설계.xlsx') as z:
 check('xlsx_zip_integrity',z.testzip() is None)
 xmls=[n for n in z.namelist() if n.endswith('.xml') or n.endswith('.rels')]
 for n in xmls:ET.fromstring(z.read(n))
 check('xlsx_xml_parse',True,str(len(xmls))+' XML documents')
 wb=ET.fromstring(z.read('xl/workbook.xml')); ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 names=[x.attrib['name'] for x in wb.findall('x:sheets/x:sheet',ns)]
 check('xlsx_sheet_names',len(set(names))==len(names) and all(len(x)<=31 for x in names),names)
# Check local links in the markdown reports; inline code is not treated as a link.
for md in H.glob('0*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',md.read_text()):
  if not link.startswith(('http:','https:','data:','#')):check(md.name+'_link_'+link,(md.parent/link).exists())
# Execute the offline report's actual JavaScript with a minimal DOM; all filter combinations.
html=(H/'종합분석보고서.html').read_text();script=re.search(r'<script>([\s\S]*?)</script>',html).group(1)
node=shutil.which('node')
check('node_available_for_filter_validation',node is not None)
harness=r'''
const vm=require('vm');let source='';process.stdin.setEncoding('utf8');process.stdin.on('data',d=>source+=d);process.stdin.on('end',()=>{
class E{constructor(value=''){this.value=value;this.children=[];this.textContent=''}add(o){this.children.push(o);if(!this.value)this.value=o.value}addEventListener(){}insertRow(){const e=new E();this.children.push(e);return e}insertCell(){const e=new E();this.children.push(e);return e}appendChild(e){this.children.push(e)}replaceChildren(...es){this.children=es}}
const es={target:new E(),split:new E('test'),dimension:new E(),results:new E()};const document={getElementById:id=>es[id],createElement:()=>new E()};
const extra=`;let count=0;for(const t of Object.keys(labels))for(const sp of ['test','validation'])for(const dim of Object.keys(dims)){target.value=t;split.value=sp;dimension.value=dim;render();const expected=rows.filter(r=>r.target===t&&r.split===sp&&r.dimension===dim);const tab=document.getElementById('results').children[0];if(tab.children.length!==expected.length+1)throw new Error('filter row mismatch');for(let i=0;i<expected.length;i++){if(String(tab.children[i+1].children[1].textContent)!==String(expected[i].n))throw new Error('N mismatch')}count++}console.log('FILTER_COMBINATIONS='+count);`;
vm.runInNewContext(source+extra,{document,Option:function(text,value){this.text=text;this.value=value},console});});
'''
r=subprocess.run([node,'-e',harness],input=script,text=True,capture_output=True)
check('offline_html_filter_execution',r.returncode==0,r.stdout+r.stderr)
check('offline_html_208_combinations','FILTER_COMBINATIONS=208' in r.stdout)
# Source provenance extended to every generator and the reused OOXML writer.
hashes=json.loads((H/'source_hashes.json').read_text())
for path in [H/'analyze.py',H/'build_report.py',Path(__file__),H.parents[1]/'report.py']:
 hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
(H/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
(H/'deliverable_checks.json').write_text(json.dumps({'passed':all(c['passed'] for c in checks),'checks':checks,
 'visual_review':'baseline_comparison.png inspected for labels/legend; no real-browser layout test, HTML filter JS tested with DOM harness',
 'created_at':datetime.datetime.now().astimezone().isoformat()},ensure_ascii=False,indent=2))
manifest={}
for path in sorted(H.rglob('*')):
 if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.zip','.pyc'] and path.name!='output_manifest.json':
  manifest[str(path.relative_to(H))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(H/'output_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
archive=H/'AX_분석산출물_20260914.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name in [*manifest,'output_manifest.json']:z.write(H/name,arcname='AX_분석산출물_20260914/'+name)
print(json.dumps({'checks_passed':len(checks),'output_files':len(manifest)+1,'zip_bytes':archive.stat().st_size,'prediction_rows':len(p)},ensure_ascii=False))
