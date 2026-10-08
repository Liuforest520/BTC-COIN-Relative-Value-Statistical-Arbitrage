import csv, math
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research'/'\u65b0\u589etradfi\u4ea4\u6613\u5bf9'
OUT=R/'\u63a8\u8350\u4ea4\u6613\u5bf9_\u4f4e\u91cd\u53e0'
BASE=OUT/'\u63a8\u8350\u4ea4\u6613\u5bf9_30\u5bf9_\u542b\u539f42\u5408\u9002pair.csv'
REVIEW=R/'\u5efa\u8bae\u4ea4\u6613\u5bf9_\u8bc4\u5ba1'/'\u8bc4\u5ba1_74\u5bf9.csv'
MET=R/'\u534f\u6574\u5206\u6790_RTH'/'42\u5bf9_\u91cd\u8dd1'/'\u6307\u6807_42\u5bf9_\u65b0\u65e7\u5bf9\u7167.csv'
DAILY=R/'\u534f\u6574\u5206\u6790_RTH'/'42\u5bf9_\u91cd\u8dd1'/'\u65e5\u7ebf_\u542b\u975e\u7f8e\u80a1.csv'
OUTFILE=OUT/'\u63a8\u8350\u4ea4\u6613\u5bf9_30\u5bf9_\u6e05\u7406\u7248.csv'; PLOTS=OUT/'\u4ef7\u683c\u8d70\u52bf\u56fe_\u6e05\u7406\u7248'
DROP={('ANET','CSCO'),('COIN','MSTR'),('IWM','TZA'),('SMH','SOXS'),('TBT','TMF')}
ADD=[('VRT','NVDA'),('SHOP','AMZN'),('MSFT','NOW'),('HOOD','COIN'),('JPM','BX')]

def plot(a,b,daily):
    ds=[];xa=[];xb=[]
    for r in daily:
        try:x,y=float(r.get(a,'')),float(r.get(b,''))
        except (ValueError,TypeError):continue
        if math.isfinite(x) and math.isfinite(y) and x>0 and y>0:ds.append(datetime.strptime(r['session_date'],'%Y-%m-%d'));xa.append(x);xb.append(y)
    xa,xb=np.asarray(xa),np.asarray(xb);na,nb=100*xa/xa[0],100*xb/xb[0];lx,ly=np.log(xa),np.log(xb);beta,alpha=np.polyfit(ly,lx,1);z=lx-(alpha+beta*ly);z=(z-z.mean())/(z.std(ddof=1) or 1)
    fig,(u,v)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1]});u.plot(ds,na,label=a,color='#1769aa');u.plot(ds,nb,label=b,color='#d97706');u.set_title(f'{a} / {b} | normalized RTH close | n={len(ds)}');u.set_ylabel('normalized close');u.grid(alpha=.25);u.legend(frameon=False,loc='upper left');v.plot(ds,z,color='#374151',lw=.9);v.axhline(2,color='#b91c1c',ls='--',lw=.8);v.axhline(-2,color='#b91c1c',ls='--',lw=.8);v.axhline(0,color='#6b7280',lw=.7);v.set_ylabel('spread z');v.grid(alpha=.25);u.xaxis.set_major_locator(mdates.YearLocator());u.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));fig.autofmt_xdate();fig.tight_layout();p=PLOTS/f'{a}_{b}.png';fig.savefig(p,dpi=140);plt.close(fig);return str(p.relative_to(OUT)).replace('\\','/')

def main():
    PLOTS.mkdir(parents=True,exist_ok=True)
    with BASE.open(encoding='utf-8-sig',newline='') as h:base=list(csv.DictReader(h))
    with REVIEW.open(encoding='utf-8-sig',newline='') as h:review={(r['\u6807\u7684A'],r['\u6807\u7684B']):r for r in csv.DictReader(h)}
    with MET.open(encoding='utf-8-sig',newline='') as h:met={(r['X'],r['Y']):r for r in csv.DictReader(h)}
    with DAILY.open(encoding='utf-8-sig',newline='') as h:daily=list(csv.DictReader(h))
    rows=[r for r in base if (r['\u6807\u7684A'],r['\u6807\u7684B']) not in DROP]
    for a,b in ADD:
        r=review[(a,b)];rows.append({'rank':'','\u6807\u7684A':a,'\u6807\u7684B':b,'\u673a\u5236':r['\u673a\u5236'],'\u903b\u8f91\u5f3a\u5ea6':r['\u903b\u8f91\u5f3a\u5ea6'],'\u516c\u5171\u6837\u672c\u5929\u6570':r['\u5929\u6570'],'EG_p':r['EG_p'],'\u53cd\u5411EG_p':r['\u53cd\u5411EG_p'],'R2':r['R2'],'betaCV':r['betaCV'],'\u534a\u8870\u671f\u5929':r['\u534a\u8870\u671f\u5929'],'absz_gt2':r['absz'],'\u586b\u5145\u7387':r['\u586b\u5145\u7387'],'\u7b49\u7ea7':'\u5206\u6563\u5316\u89c2\u5bdf','\u91d1\u878d\u903b\u8f91':r['\u7528\u6237\u7ed9\u7684\u7ecf\u6d4e\u903b\u8f91'],'\u9009\u62e9\u539f\u56e0':'\u4e3a\u907f\u514d\u539f\u63a8\u8350\u4e2d\u7684\u4f4e\u76f8\u5173\u6216\u53cd\u5411\u6760\u6746\u7ed3\u6784\uff0c\u6539\u7528\u6b63\u76f8\u5173\u4e14\u4ef7\u683c\u5f62\u6001\u66f4\u4e00\u81f4\u7684\u5019\u9009','\u98ce\u9669\u63d0\u793a':'\u7ed9\u5b9a\u65b9\u5411 EG \u6216\u56de\u5f52\u901f\u5ea6\u4e0d\u662f\u6838\u5fc3\u7ea7\uff0c\u4f7f\u7528\u524d\u9700\u505a\u6eda\u52a8\u7a33\u5b9a\u6027\u68c0\u9a8c','\u4ef7\u683c\u8d70\u52bf\u56fe':'','\u6807\u7684A\u51fa\u73b0\u6b21\u6570':'','\u6807\u7684B\u51fa\u73b0\u6b21\u6570':''})
    usage={};
    for r in rows:
        key=(r['\u6807\u7684A'],r['\u6807\u7684B'])
        if key in review: r['corr']=review[key]['corr']
        elif key in met: r['corr']=met[key]['\u65b0_\u65e5\u6536\u76ca\u76f8\u5173']
        else: r['corr']=''
    for r in rows:usage[r['\u6807\u7684A']]=usage.get(r['\u6807\u7684A'],0)+1;usage[r['\u6807\u7684B']]=usage.get(r['\u6807\u7684B'],0)+1
    for i,r in enumerate(rows,1):
        r['rank']=i;r['\u6807\u7684A\u51fa\u73b0\u6b21\u6570']=usage[r['\u6807\u7684A']];r['\u6807\u7684B\u51fa\u73b0\u6b21\u6570']=usage[r['\u6807\u7684B']];r['\u4ef7\u683c\u8d70\u52bf\u56fe']=plot(r['\u6807\u7684A'],r['\u6807\u7684B'],daily)
    with OUTFILE.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'\u6807\u7684\u4f7f\u7528\u6b21\u6570_30\u5bf9_\u6e05\u7406\u7248.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.writer(h);w.writerow(['\u6807\u7684','\u4f7f\u7528\u6b21\u6570']);w.writerows(sorted(usage.items(),key=lambda x:(-x[1],x[0])))
    print(f'rows={len(rows)} unique_symbols={len(usage)} max_usage={max(usage.values())} output={OUTFILE}')
if __name__=='__main__':main()
