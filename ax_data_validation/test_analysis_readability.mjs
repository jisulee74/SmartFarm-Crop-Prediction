import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=process.cwd()+'/ax_data_validation';
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{id,innerHTML:'',textContent:'',style:{},classList:{add(){},remove(){},toggle(){}},querySelectorAll(){return[];},addEventListener(){},append(){},value:''});return elements.get(id);};
globalThis.document={getElementById:element,querySelectorAll(){return[];},querySelector(){return null;},createElement:element};
const charts=[];
globalThis.echarts={init(){const c={option:null,setOption(v){this.option=v;},clear(){this.option=null;},resize(){},on(){},off(){}};charts.push(c);return c;}};
globalThis.window={echarts:globalThis.echarts,addEventListener(){}};
const {AnalysisViewController}=await import(`file://${root}/src/analysis_view.js`);
const v=new AnalysisViewController();v.bundle=JSON.parse(await fs.readFile(root+'/ax_direction/dashboard_bundle.json','utf8'));
v.chartModelComp=echarts.init();v.chartErrorStrata=echarts.init();v.chartSseShare=echarts.init();
for(const target of v.bundle.tables.selected_models.map(r=>r.target)){
 for(const split of ['test','validation','test']){
  v.state.target=target;v.state.split=split;v.renderTargetDependentViews();v.renderFullKpiTable();
  assert.ok(!element('anl_target_kpi_grid').innerHTML.includes('NaN'));
  assert.ok(!element('anl_summary_rmse').textContent.includes('NaN'));
  if(split==='validation'){
   assert.equal(v.chartErrorStrata.option,null);
   assert.equal(v.chartSseShare.option,null);
   assert.ok(element('anl_crossed_tbody').innerHTML.includes('미제공'));
   assert.ok(element('anl_diagnosis_content').innerHTML.includes('저장하지 않아'));
  }else{
   assert.ok(v.chartErrorStrata.option);
   assert.ok(!element('anl_crossed_tbody').innerHTML.includes('미제공'));
  }
 }
}
const { DATASET_AUDIT } = await import(`file://${root}/src/dataset_audit.js`);
for (const [target, audit] of Object.entries(DATASET_AUDIT.targets)) {
 v.state.target=target;
 for (const split of ['test','validation']) {
  v.state.split=split; v.renderTargetDependentViews();
  const table=element('anl_dataset_composition').innerHTML;
  assert.ok(table.includes(audit.splits.train.feature_date_start));
  assert.ok(table.includes(audit.splits.test.target_date_end));
  assert.ok(table.includes(audit.splits.test.n.toLocaleString()+'건'));
  const html=element('anl_dist_warning_card').innerHTML;
  assert.ok(html.includes('AX 보완 제안 · 효과 검증 전'));
  if(split==='test') assert.ok(html.includes((audit.test.zero_error_share*100).toFixed(1)+'%'));
  else assert.ok(html.includes('행별 예측은 저장되지 않아'));
 }
}
v.bundle.metadata.source_campaign='future_campaign';
v.renderTargetDependentViews();
assert.ok(element('anl_dataset_composition').textContent.includes('확인 중'));
assert.ok(!element('anl_dist_warning_card').innerHTML.includes('전체 제곱오차의'));
console.log('PASS: all 8 targets, Test→Validation→Test; no stale strata, chart reset, KPI rendering');
