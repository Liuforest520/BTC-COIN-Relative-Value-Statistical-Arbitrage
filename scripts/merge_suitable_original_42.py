import csv, math
from collections import Counter
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
BASE=OUT/'\u63a8\u8350\u4ea4\u6613\u5bf9_25\u5bf9.csv'
MET=R/'\u534f\u6574\u5206\u6790_RTH'/'42\u5bf9_\u91cd\u8dd1'/'\u6307\u6807_42\u5bf9_\u65b0\u65e7\u5bf9\u7167.csv'
DAILY=R/'\u534f\u6574\u5206\u6790_RTH'/'42\u5bf9_\u91cd\u8dd1'/'\u65e5\u7ebf_\u542b\u975e\u7f8e\u80a1.csv'
PLOTS=OUT/'\u4ef7\u683c\u8d70\u52bf\u56fe_30\u5bf9'

EXTRA={
 ('BTC','COIN'):('L3 \u540c\u4e00\u7ec8\u7aef\u9700\u6c42','4','BTC\u4ef7\u683c\u4e0a\u6da8\u901a\u5e38\u63d0\u5347\u52a0\u5bc6\u5e02\u573a\u4ea4\u6613\u6d3b\u8dc3\u5ea6\uff0c\u8fdb\u800c\u5f71\u54cd Coinbase \u4ea4\u6613\u91cf\u3001\u624b\u7eed\u8d39\u6536\u5165\u548c\u4f30\u503c\u3002'),
 ('XAU','XPT'):('L3 \u540c\u4e00\u7ec8\u7aef\u9700\u6c42','4','\u9ec4\u91d1\u548c\u94c2\u91d1\u5171\u4eab\u5b8f\u89c2\u3001\u5229\u7387\u3001\u7f8e\u5143\u548c\u8d35\u91d1\u5c5e\u5546\u54c1\u5468\u671f\u9a71\u52a8\u3002'),
 ('XAG','XPT'):('L3 \u540c\u4e00\u7ec8\u7aef\u9700\u6c42','4','\u767d\u94f6\u548c\u94c2\u91d1\u90fd\u53d7\u8d35\u91d1\u5c5e\u5b8f\u89c2\u56e0\u5b50\u548c\u5de5\u4e1a\u9700\u6c42\u5f71\u54cd\u3002'),
 ('XAU','XAG'):('L3 \u540c\u4e00\u7ec8\u7aef\u9700\u6c42','4','\u91d1\u94f6\u6bd4\u662f\u4f20\u7edf\u8d35\u91d1\u5c5e\u76f8\u5bf9\u4ef7\u503c\u5173\u7cfb\uff0c\u5171\u540c\u53d7\u5b9e\u9645\u5229\u7387\u3001\u7f8e\u5143\u548c\u907f\u9669\u9700\u6c42\u9a71\u52a8\u3002'),
 ('AMD','ARM'):('L3 \u76f4\u63a5\u7ade\u54c1','4','AMD \u548c ARM \u90fd\u53d7 CPU\u3001\u6570\u636e\u4e2d\u5fc3\u3001AI \u548c\u534a\u5bfc\u4f53\u5468\u671f\u5f71\u54cd\uff0c\u4f46\u6536\u5165\u6a21\u5f0f\u5e76\u975e\u5b8c\u5168\u76f8\u540c\u3002')}

def num(x):
    try:return float(x)
    except (ValueError,TypeError):return math.nan

def plot(a,b,daily):
    ds=[]; xa=[]; xb=[]
    for r in daily:
        try:x,y=float(r.get(a,'')),float(r.get(b,''))
        except (ValueError,TypeError):continue
        if math.isfinite(x) and math.isfinite(y) and x>0 and y>0:
            ds.append(datetime.strptime(r['session_date'],'%Y-%m-%d')); xa.append(x); xb.append(y)
    if len(ds)<2:return ''
    xa,xb=np.asarray(xa),np.asarray(xb); na,nb=100*xa/xa[0],100*xb/xb[0]
    lx,ly=np.log(xa),np.log(xb); beta,alpha=np.polyfit(ly,lx,1); z=lx-(alpha+beta*ly); z=(z-z.mean())/(z.std(ddof=1) or 1)
    fig,(u,v)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    u.plot(ds,na,label=a,color='#1769aa',lw=1.1);u.plot(ds,nb,label=b,color='#d97706',lw=1.1);u.set_title(f'{a} / {b} | normalized RTH close | n={len(ds)}');u.set_ylabel('normalized close');u.grid(alpha=.25);u.legend(frameon=False,loc='upper left')
    v.plot(ds,z,color='#374151',lw=.9);v.axhline(2,color='#b91c1c',ls='--',lw=.8);v.axhline(-2,color='#b91c1c',ls='--',lw=.8);v.axhline(0,color='#6b7280',lw=.7);v.set_ylabel('spread z');v.grid(alpha=.25)
    u.xaxis.set_major_locator(mdates.YearLocator());u.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));fig.autofmt_xdate();fig.tight_layout();p=PLOTS/f'{a}_{b}.png';fig.savefig(p,dpi=140);plt.close(fig);return str(p.relative_to(OUT)).replace('\\','/')

def main():
    PLOTS.mkdir(parents=True,exist_ok=True)
    with BASE.open(encoding='utf-8-sig',newline='') as h: base=list(csv.DictReader(h))
    with MET.open(encoding='utf-8-sig',newline='') as h: metrics={(r['X'],r['Y']):r for r in csv.DictReader(h)}
    with DAILY.open(encoding='utf-8-sig',newline='') as h: daily=list(csv.DictReader(h))
    existing={(r['\u6807\u7684A'],r['\u6807\u7684B']) for r in base}; usage=Counter()
    for r in base: usage[r['\u6807\u7684A']]+=1; usage[r['\u6807\u7684B']]+=1
    rows=list(base); start=len(rows)+1
    for (a,b),(mechanism,logic,reason) in EXTRA.items():
        if (a,b) in existing: continue
        m=metrics[(a,b)]; fill='0.0'; p=float(m['\u65b0_EG_p']); r2=float(m['\u65b0_R2']); bcv=float(m['\u65b0_betaCV']); half=float(m['\u65b0_\u534a\u8870\u671f\u5929'])
        grade='\u6838\u5fc3' if p<.05 and r2>=.7 and bcv<=1.2 and float(m['\u4ea4\u6613\u65e5'])>=500 else '\u4f18\u5148\u5907\u9009'
        rows.append({'rank':start,'\u6807\u7684A':a,'\u6807\u7684B':b,'\u673a\u5236':mechanism,'\u903b\u8f91\u5f3a\u5ea6':logic,'\u516c\u5171\u6837\u672c\u5929\u6570':m['\u4ea4\u6613\u65e5'],'EG_p':m['\u65b0_EG_p'],'\u53cd\u5411EG_p':'','R2':m['\u65b0_R2'],'betaCV':m['\u65b0_betaCV'],'\u534a\u8870\u671f\u5929':m['\u65b0_\u534a\u8870\u671f\u5929'],'absz_gt2':m['\u65b0_|z|>2\u5360\u6bd4'],'\u586b\u5145\u7387':fill,'\u7b49\u7ea7':grade,'\u91d1\u878d\u903b\u8f91':reason,'\u9009\u62e9\u539f\u56e0':f'\u539f42\u5bf9\u91cd\u65b0\u68c0\u9a8c\uff1aR2={r2}\uff0cEG p={p}\uff0c\u534a\u8870\u671f={half}\u5929\uff0c\u53ea\u56e0\u6837\u672c\u6216\u7edf\u8ba1\u53e3\u5f84\u4f5c\u4fdd\u7559\u8bf4\u660e\u89c1\u5907\u6ce8','\u98ce\u9669\u63d0\u793a':'\u539f42\u5bf9\u65b0\u589e\uff1a\u90e8\u5206\u975e\u7f8e\u80a1\u6837\u672c\u504f\u77ed' if float(m['\u4ea4\u6613\u65e5'])<300 else '\u53cd\u5411EG\u672a\u5728\u539f42\u5bf9\u8868\u4e2d\u63d0\u4f9b','\u4ef7\u683c\u8d70\u52bf\u56fe':plot(a,b,daily),'\u6807\u7684A\u51fa\u73b0\u6b21\u6570':'','\u6807\u7684B\u51fa\u73b0\u6b21\u6570':''})
        usage[a]+=1; usage[b]+=1; start+=1
    for r in rows: r['\u6807\u7684A\u51fa\u73b0\u6b21\u6570']=usage[r['\u6807\u7684A']]; r['\u6807\u7684B\u51fa\u73b0\u6b21\u6570']=usage[r['\u6807\u7684B']]
    out=OUT/'\u63a8\u8350\u4ea4\u6613\u5bf9_30\u5bf9_\u542b\u539f42\u5408\u9002pair.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'\u6807\u7684\u4f7f\u7528\u6b21\u6570_30\u5bf9.csv').open('w',encoding='utf-8-sig',newline='') as h: w=csv.writer(h);w.writerow(['\u6807\u7684','\u4f7f\u7528\u6b21\u6570']);w.writerows(sorted(usage.items(),key=lambda x:(-x[1],x[0])))
    print(f'rows={len(rows)} unique_symbols={len(usage)} max_usage={max(usage.values())} output={out}')

if __name__=='__main__':main()
