"""Read original growth serials and explicit crop links; no database writes."""
from pathlib import Path
import os
import pandas as pd
import pymysql
from .catalog import AX


def main():
    cfg={}
    env=Path('GEAS/GEAS3.5/.env')
    if env.exists():
        for line in env.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('\"\'')
    def setting(k,default=None):return os.getenv('FARMSTOM_DB_'+k) or cfg.get('GEAS_DB_'+k,default)
    raw=pd.read_parquet(AX/'tomato_flower_count_prediction/data/raw/growth.parquet')
    ids=sorted(raw.source_sn.astype(int).unique().tolist());parts=[]
    c=pymysql.connect(host=setting('HOST'),port=int(setting('PORT',3306)),user=setting('USER'),password=setting('PASSWORD'),database='farmstom_usoo',charset='utf8mb4',connect_timeout=10,read_timeout=120)
    try:
        with c.cursor() as q:
            q.execute('SET TRANSACTION READ ONLY');q.execute('START TRANSACTION')
            for i in range(0,len(ids),1000):
                batch=ids[i:i+1000]
                q.execute('SELECT g.sn AS source_sn,g.cropping_number,c.sn AS matched_crop_sn,c.cropping_serl_no,c.cropping_date,c.cropping_end_date FROM sfkr_hbfm_grow g LEFT JOIN sfkr_pvsn_crop c ON g.cropping_number=c.cropping_serl_no AND SUBSTRING_INDEX(g.facility_id,\'_\',2)=c.user_id AND g.item_code=c.item_code WHERE g.sn IN ('+','.join(['%s']*len(batch))+')',batch)
                parts.append(pd.DataFrame(q.fetchall(),columns=[x[0] for x in q.description]))
    finally:c.rollback();c.close()
    fetched=pd.concat(parts,ignore_index=True)
    if fetched.source_sn.duplicated().any():raise ValueError('Multiple crop matches for source row')
    if set(fetched.source_sn)!=set(ids):raise ValueError('Source observations missing in live database')
    joined=raw[['source_sn','facility_id','crop_sn','examin_date']].merge(fetched,on='source_sn',validate='many_to_one')
    joined=joined.rename(columns={'examin_date':'timestamp','cropping_date':'crop_start_date','cropping_end_date':'crop_end_date'})
    keys=['facility_id','crop_sn','timestamp'];joined=joined.drop(columns='source_sn').drop_duplicates()
    if joined.duplicated(keys).any():raise ValueError('Trusses disagree on direct crop assignment')
    for col in ['timestamp','crop_start_date','crop_end_date']:joined[col]=pd.to_datetime(joined[col],errors='raise').dt.normalize()
    if joined.matched_crop_sn.isna().any():raise ValueError('Unmatched explicit crop number')
    if not (joined.timestamp.ge(joined.crop_start_date)&joined.timestamp.le(joined.crop_end_date)).all():raise ValueError('Recorded observation outside directly matched crop period')
    joined['crop_match_method']='cropping_number_to_cropping_serl_no_user_item_verified'
    joined['crop_match_status']='matched';joined['crop_match_count']=1
    out=Path(__file__).parent/'outputs/source_snapshot';out.mkdir(parents=True,exist_ok=True)
    joined.to_parquet(out/'tomato_direct_crop_dates.parquet',index=False)
    print('Live source rows:',len(ids),'verified observation/cycle links:',len(joined))

if __name__=='__main__':main()
