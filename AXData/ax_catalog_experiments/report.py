from __future__ import annotations
import json
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from .catalog import AX,build_catalog
from .core import digest,freeze
from .data import TARGETS


def write_xlsx(path,tables):
    """Dependency-free OOXML writer; strings never become spreadsheet formulas."""
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    def col(n):
        s=''
        while n:n,r=divmod(n-1,26);s=chr(65+r)+s
        return s
    def clean(v):return ''.join(c for c in str(v) if ord(c)>=32 or c in '\n\t\r')[:32767]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        overrides=''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(tables)+1))
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+overrides+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        sheets=''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(tables,1))
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheets}</sheets></workbook>')
        rel=''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(tables)+1))
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+rel+'</Relationships>')
        for i,(name,frame) in enumerate(tables.items(),1):
            if len(frame)>1048575:raise ValueError(f'Excel row limit: {name}; use Parquet')
            rows=[]
            for n,values in enumerate([frame.columns.tolist(),*frame.itertuples(index=False,name=None)],1):
                cells=[]
                for j,v in enumerate(values,1):
                    ref=f'{col(j)}{n}'
                    if v is None or (isinstance(v,float) and not math.isfinite(v)):continue
                    if isinstance(v,(float,int)) and not isinstance(v,bool):cells.append(f'<c r="{ref}"><v>{v}</v></c>')
                    else:cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(clean(v))}</t></is></c>')
                rows.append(f'<row r="{n}">'+''.join(cells)+'</row>')
            width=max(1,len(frame.columns));last=max(1,len(frame)+1)
            xml=f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>'+''.join(rows)+f'</sheetData><autoFilter ref="A1:{col(width)}{last}"/></worksheet>'
            z.writestr(f'xl/worksheets/sheet{i}.xml',xml)


def legacy_records():
    records=[];definitions=[]
    tomato=AX/'tomato_flower_count_prediction/artifacts/total_flower_count_v1_sample_key_retry3/model_selection'
    strawberry=AX/'strawberry_fruit_set_prediction/facility_holdout_fixed13_with_tft_v1'
    def add(target,group,model,split,metrics,path,features,seed_count=None):
        rid='LEGACY_'+digest([target,group,model,split,str(path)])[:16]
        records.append({'Run_ID':rid,'Target_ID':target,'Variable_Group_ID':group,'Model_Type':model,'Seed':None,'Seed_Count':seed_count,'Stage':split,'Status':'success','Result_Origin':'legacy','Aggregation':'source_reported_unknown_or_ensemble','Source_Path':str(path),'Metrics':{k:metrics[k] for k in ('rmse','mae','r2','ccc') if k in metrics},'Features':features,'Categories':['LEGACY_CAT_'+group],'Feature_Count':len(features),'N_Eval':None,'Eval_Row_Hash':'legacy_unverified'})
    config=json.loads((tomato/'selected_configuration.json').read_text());features=config['features']
    for r in pd.read_csv(tomato/'validation/model_metrics.csv').to_dict('records'):add('tomato_total_flower','FlowerA_legacy',r['model'],'validation',r,tomato/'validation/model_metrics.csv',features,r.get('seed_count'))
    for model,file in [(config['selected_learning_model'],'final_test_metrics.json'),('persistence','persistence_test_metrics.json')]:
        add('tomato_total_flower','FlowerA_legacy',model,'test',json.loads((tomato/file).read_text()),tomato/file,features)
    for split,file in [('validation','validation_all_model_metrics.csv'),('test','final_test_metrics.csv')]:
        p=strawberry/'summary'/file
        for r in pd.read_csv(p).to_dict('records'):
            truss=r['truss'];target='strawberry_'+truss.split('_')[0]
            f=json.loads((strawberry/'artifacts'/truss/'selected_configuration.json').read_text())['features']
            add(target,'FruitA_legacy',r['model'],split,r,p,f,r.get('seed_count'))
    return records


EMPTY={
 'Experiments':['Experiment_ID','Target_ID','Crop_Target','Prediction_Target','Variable_Group_ID','Dataset_ID','Split_ID','Result_Origin'],
 'Runs':['Run_ID','Experiment_ID','Model_Type','Seed','Stage','Status','Error','Input_Policy'],
 'Results':['Run_ID','Experiment_ID','Crop_Target','Prediction_Target','Variable_Group_ID','Model_Type','Seed','Eval_Split','Metric_Name','Metric_Value','N_Eval','Result_Origin','Eval_Row_Hash'],
 'Best_Configurations':['Target_ID','Model_Type','Variable_Group_ID','Validation_RMSE','Overall_Selected'],
 'Category_Comparisons':['Target_ID','Model_Type','Base_Group_ID','Added_Group_ID','Category_ID','RMSE_Difference','Eval_Row_Hash'],
 'Availability':['Target_ID','Category_ID','Variable_ID','Status','Reason'],
 'Variable_Groups':['Variable_Group_ID','Category_ID'],
 'Group_Detail':['Variable_Group_ID','Category_ID','Variable_ID','Variable_Name'],
}


def build_report(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source,variables,categories,members=build_catalog()
    freeze(output/'catalog_definition.json',{'sources':source.to_dict('records'),'variables':variables.to_dict('records'),'categories':categories.to_dict('records'),'members':members.to_dict('records')})
    tables={'README':pd.DataFrame([{'Item':'Scope','Value':'4 targets; category atomic selection; legacy is not comparable with new runs'},{'Item':'DB inventory','Value':'NOT COMPLETE: local schema/code evidence only unless reviewed DB inventory is supplied'},{'Item':'Metrics','Value':'Long format; never sum metric values or mix targets/stages/origins'},{'Item':'New runs','Value':'Smoke is verification only. New full experiments require prepared raw provenance and engine full completion.'}]),'Source_Inventory':source,'Variable_Catalog':variables,'Category_Catalog':categories,'Category_Variables':members}
    snapshot=output/'source_snapshot/db_schema.csv'
    if snapshot.exists():
        db=pd.read_csv(snapshot).fillna('')
        db=db.rename(columns={'table_name':'Source_Table','column_name':'Source_Field_or_Code','column_comment':'Description','column_type':'Data_Type'})
        db['Evidence']='live_db_schema_snapshot';db['Status']='needs_value_and_join_review'
        tables['Source_Inventory']=pd.concat([source,db],ignore_index=True).drop_duplicates(['Source_Table','Source_Field_or_Code'],keep='last')
    records=legacy_records();specs={};availability=[];group_rows=[];member_rows=[];category_rows=[];extra_variables=[]
    for p in (output/'prepared').glob('*/manifest.json'):
        spec=json.loads(p.read_text());specs[spec['target_id']]=spec
        defined={f for fs in spec['categories'].values() for f in fs}
        freeze(p.parent/'variable_definitions.json',variables[variables.Variable_ID.isin(defined)].to_dict('records'))
        availability.append(pd.read_csv(p.parent/'availability.csv'))
        c=pd.read_csv(p.parent/'categories.csv')
        for r in c.to_dict('records'):
            original=categories[categories.Category_ID.eq(r['Source_Category_ID'])].iloc[0].to_dict()
            category_rows.append({**original,**r})
        for cid,fs in spec['categories'].items():member_rows.extend({'Category_ID':cid,'Variable_ID':f} for f in fs)
    for p in (output/'experiments').glob('**/runs/*/result.json'):records.append(json.loads(p.read_text()))
    legacy_seen=set()
    for r in records:
        gid=r['Variable_Group_ID']
        for c in r.get('Categories',[]):group_rows.append({'Variable_Group_ID':gid,'Category_ID':c})
        if r['Result_Origin']=='legacy' and gid not in legacy_seen:
            legacy_seen.add(gid);cid='LEGACY_CAT_'+gid
            category_rows.append({'Category_ID':cid,'AX_Category':'Mixed','Variable_Domain':'Legacy','Variable_Category':gid+' 原본 입력 목록','Version':1,'Status':'legacy_preserved'})
            for f in r['Features']:
                vid=gid+'::'+f
                extra_variables.append({'Variable_ID':vid,'Variable_Name':f,'Source_Table':'retained legacy artifacts','Source_Field_or_Code':f,'Category_ID':cid,'Availability_Status':'legacy_definition_not_rebuilt','Definition_Version':1})
                member_rows.append({'Category_ID':cid,'Variable_ID':vid})
    tables['Variable_Catalog']=pd.concat([variables,pd.DataFrame(extra_variables)],ignore_index=True)
    tables['Category_Catalog']=pd.concat([categories,pd.DataFrame(category_rows)],ignore_index=True).drop_duplicates('Category_ID')
    tables['Category_Variables']=pd.concat([members,pd.DataFrame(member_rows)],ignore_index=True).drop_duplicates()
    tables['Variable_Groups']=pd.DataFrame(group_rows,columns=EMPTY['Variable_Groups']).drop_duplicates()
    tables['Group_Detail']=tables['Variable_Groups'].merge(tables['Category_Variables'],on='Category_ID').merge(tables['Variable_Catalog'].drop(columns=['Category_ID']),on='Variable_ID',validate='many_to_one').merge(tables['Category_Catalog'],on='Category_ID',validate='many_to_one')
    tables['Availability']=pd.concat(availability,ignore_index=True) if availability else pd.DataFrame(columns=EMPTY['Availability'])
    experiments={};runs=[];results=[]
    for r in records:
        target=r['Target_ID'];spec=specs.get(target,{}) if r['Result_Origin']!='legacy' else {}
        gid=r['Variable_Group_ID'];eid='EXP_'+digest([target,gid,r['Result_Origin'],spec.get('dataset_id'),spec.get('split_id')])[:16]
        crop,label=TARGETS[target]
        experiments[eid]={'Experiment_ID':eid,'Target_ID':target,'Crop_Target':crop,'Prediction_Target':label,'Variable_Group_ID':gid,'Dataset_ID':spec.get('dataset_id','legacy_unverified'),'Split_ID':spec.get('split_id','legacy_unverified'),'Result_Origin':r['Result_Origin'],'Prediction_Horizon':'next survey'}
        run={k:v for k,v in r.items() if k not in {'Metrics','Categories','Features','Params'}}
        run.update(Experiment_ID=eid,Params_JSON=json.dumps(r.get('Params',{}),sort_keys=True),Input_Policy='baseline_current_target_only' if r['Model_Type']=='persistence' else 'legacy_source_definition' if r['Result_Origin']=='legacy' else 'all_frozen_category_members')
        runs.append(run)
        for metric,value in r.get('Metrics',{}).items():
            results.append({'Run_ID':r['Run_ID'],'Experiment_ID':eid,'Crop_Target':crop,'Prediction_Target':label,'Variable_Group_ID':gid,'Model_Type':r['Model_Type'],'Seed':r.get('Seed'),'Eval_Split':r['Stage'],'Fold_ID':r.get('Fold_ID'),'Metric_Name':metric.upper(),'Metric_Value':value,'Metric_Status':'undefined' if value is None else 'available','N_Eval':r.get('N_Eval'),'Result_Origin':r['Result_Origin'],'Eval_Row_Hash':r.get('Eval_Row_Hash'),'Aggregation':r.get('Aggregation','per_seed')})
    tables['Experiments']=pd.DataFrame(experiments.values(),columns=EMPTY['Experiments']+['Prediction_Horizon'])
    tables['Runs']=pd.DataFrame(runs);tables['Results']=pd.DataFrame(results)
    tables['Execution_Progress']=tables['Runs'].groupby(['Target_ID','Result_Origin','Stage','Model_Type','Status'],dropna=False).size().rename('Completed_Run_Count').reset_index()
    best=[]
    for p in (output/'experiments').glob('*/full/selected.json'):
        obj=json.loads(p.read_text());target=p.parents[1].name
        for model,w in obj['winners'].items():
            gid='VG_'+digest({'categories':sorted(w['group']),'features':sorted({f for c in w['group'] for f in specs[target]['categories'][c]})})[:16]
            best.append({'Target_ID':target,'Model_Type':model,'Variable_Group_ID':gid,'Validation_RMSE':w['validation_rmse'],'Overall_Selected':model==obj['overall']})
    tables['Best_Configurations']=pd.DataFrame(best,columns=EMPTY['Best_Configurations'])
    per_seed=tables['Results'][tables['Results'].Result_Origin.eq('new') & tables['Results'].Eval_Split.isin(['validation','test'])]
    keys=['Experiment_ID','Variable_Group_ID','Model_Type','Eval_Split','Metric_Name','Eval_Row_Hash']
    tables['Results_Summary']=per_seed.groupby(keys,dropna=False).Metric_Value.agg(Mean='mean',Std='std',Seed_Count='count').reset_index()
    comparisons=[]
    valid=[r for r in records if r['Result_Origin']=='new' and r['Stage']=='validation' and r['Status']=='success']
    for a in valid:
        for b in valid:
            if (a['Target_ID'],a['Model_Type'],a['Seed'],a['Eval_Row_Hash'])!=(b['Target_ID'],b['Model_Type'],b['Seed'],b['Eval_Row_Hash']):continue
            ca,cb=set(a['Categories']),set(b['Categories'])
            if ca<cb and len(cb-ca)==1:
                comparisons.append({'Target_ID':a['Target_ID'],'Model_Type':a['Model_Type'],'Seed':a['Seed'],'Base_Group_ID':a['Variable_Group_ID'],'Added_Group_ID':b['Variable_Group_ID'],'Category_ID':next(iter(cb-ca)),'RMSE_Difference':b['Metrics']['rmse']-a['Metrics']['rmse'],'Eval_Row_Hash':a['Eval_Row_Hash'],'Interpretation':'association after per-group tuning, not causal effect'})
    tables['Category_Comparisons']=pd.DataFrame(comparisons) if comparisons else pd.DataFrame(columns=EMPTY['Category_Comparisons'])
    campaign=output/'campaign_status.json'
    if campaign.exists():tables['Campaign_Status']=pd.DataFrame(json.loads(campaign.read_text()))
    status=output/'prepared/target_status.csv'
    if status.exists():tables['Target_Status']=pd.read_csv(status)
    # Explicit uniqueness and foreign-key checks before exporting.
    for name,key in [('Variable_Catalog','Variable_ID'),('Category_Catalog','Category_ID'),('Experiments','Experiment_ID'),('Runs','Run_ID')]:
        if tables[name][key].duplicated().any():raise ValueError(f'Duplicate {name}.{key}')
    if not set(tables['Results'].Run_ID)<=set(tables['Runs'].Run_ID):raise ValueError('Orphan result')
    from .display_ids import simplify_ids
    simplify_ids(tables,output)
    csv=output/'csv';csv.mkdir(exist_ok=True)
    for name,frame in tables.items():frame.to_csv(csv/f'{name}.csv',index=False,encoding='utf-8-sig')
    write_xlsx(output/'AX_Experiment_Report.xlsx',tables)
    summary={name:len(frame) for name,frame in tables.items()}
    (output/'report_manifest.json').write_text(json.dumps({'db_inventory_complete':False,'row_counts':summary},ensure_ascii=False,indent=2))
    (output/'Findings.md').write_text('# 결과 상태\n\nDB 전체 모집단은 아직 확정되지 않았습니다. 원천 스키마·코드 기반 후보 사전과 과거 결과를 수입했습니다.\n\nLegacy 결과는 새 범주 조합과 직접 비교하지 않습니다. smoke_not_rankable은 실행 검증이며 최종 모델 선정 결과가 아닙니다. 신규 완료 여부는 experiments/*/full/status.json을 확인합니다.\n\n'+ '\n'.join(f'- {k}: {v}행' for k,v in summary.items())+'\n',encoding='utf-8')
    return summary
