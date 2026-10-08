import csv, math
from datetime import datetime
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'research' / '\u65b0\u589etradfi\u4ea4\u6613\u5bf9'
REVIEW = R / '\u5efa\u8bae\u4ea4\u6613\u5bf9_\u8bc4\u5ba1' / '\u8bc4\u5ba1_74\u5bf9.csv'
DAILY = R / '\u534f\u6574\u5206\u6790_RTH' / '42\u5bf9_\u91cd\u8dd1' / '\u65e5\u7ebf_\u542b\u975e\u7f8e\u80a1.csv'
OUT = R / '\u63a8\u8350\u4ea4\u6613\u5bf9_\u4f4e\u91cd\u53e0'
PLOTS = OUT / '\u4ef7\u683c\u8d70\u52bf\u56fe'
TARGET = 25

def num(row, key, default=math.nan):
    try:
        return float(row.get(key, ''))
    except (TypeError, ValueError):
        return default

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def excluded(r):
    return (num(r, '\u5929\u6570') < 100 or num(r, '\u903b\u8f91\u5f3a\u5ea6') < 2
            or num(r, 'R2') < .20 or num(r, 'betaCV') > 2.50
            or num(r, '\u534a\u8870\u671f\u5929') > 400 or num(r, '\u586b\u5145\u7387') > .25)

def score(r):
    p = max(num(r, 'EG_p', 1), 1e-6)
    rp = max(num(r, '\u53cd\u5411EG_p', 1), 1e-6)
    r2, bcv = num(r, 'R2', 0), num(r, 'betaCV', 99)
    half, days = num(r, '\u534a\u8870\u671f\u5929', 9999), num(r, '\u5929\u6570', 0)
    z, fill, logic = num(r, 'absz', 0), num(r, '\u586b\u5145\u7387', 1), num(r, '\u903b\u8f91\u5f3a\u5ea6', 0)
    return (1.8 * logic + 1.5 * clip(-math.log10(p), 0, 6)
            + clip(-math.log10(max(p, rp)), 0, 6) + 1.4 * clip(r2, 0, 1)
            + .9 * clip(1.2 - bcv, -1.5, 1.2) + .8 * clip(1 - half / 180, -2, 1)
            + .5 * clip(math.log10(max(days, 100) / 100), 0, 1.2)
            + .4 * clip(z / .05, 0, 1.5) + .3 * clip(1 - fill / .25, -1, 1))

def grade(r):
    p, rp, r2, bcv = num(r, 'EG_p'), num(r, '\u53cd\u5411EG_p'), num(r, 'R2'), num(r, 'betaCV')
    if p <= .10 and rp <= .20 and r2 >= .70 and bcv <= 1.20:
        return '\u6838\u5fc3'
    if p <= .25 and rp <= .40 and r2 >= .50 and bcv <= 1.50:
        return '\u4f18\u5148\u5907\u9009'
    return '\u5206\u6563\u5316\u89c2\u5bdf'

def risk(r):
    out = []
    if num(r, 'EG_p') > .05: out.append('\u7ed9\u5b9a\u65b9\u5411EG\u672a\u8fbe5%\u663e\u8457')
    if num(r, '\u53cd\u5411EG_p') > .20: out.append('\u53cd\u5411EG\u504f\u5f31')
    if num(r, 'R2') < .70: out.append('R2\u4e00\u822c')
    if num(r, 'betaCV') > 1.20: out.append('beta\u7a33\u5b9a\u6027\u4e00\u822c')
    if num(r, '\u534a\u8870\u671f\u5929') > 90: out.append('\u534a\u8870\u671f\u504f\u957f')
    if num(r, '\u5929\u6570') < 250: out.append('\u6837\u672c\u504f\u77ed')
    if num(r, '\u586b\u5145\u7387') > .10: out.append('\u586b\u5145\u7387\u504f\u9ad8')
    if '\u6760\u6746' in r.get('\u673a\u5236', ''): out.append('\u6760\u6746ETF\u6709\u590d\u4f4d/\u6ce2\u52a8\u635f\u8017\u98ce\u9669')
    return '\uff1b'.join(out) or '\u7edf\u8ba1\u548c\u6570\u636e\u8d28\u91cf\u5747\u8f83\u597d'

def plot_pair(r, daily):
    a, b = r['\u6807\u7684A'], r['\u6807\u7684B']
    dates, xa, xb = [], [], []
    for row in daily:
        try: x, y = float(row.get(a, '')), float(row.get(b, ''))
        except (TypeError, ValueError): continue
        if math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0:
            dates.append(datetime.strptime(row['session_date'], '%Y-%m-%d')); xa.append(x); xb.append(y)
    if len(dates) < 2: return ''
    xa, xb = np.asarray(xa), np.asarray(xb)
    na, nb = 100 * xa / xa[0], 100 * xb / xb[0]
    lx, ly = np.log(xa), np.log(xb)
    beta, alpha = np.polyfit(ly, lx, 1)
    z = lx - (alpha + beta * ly); z = (z - z.mean()) / (z.std(ddof=1) or 1)
    fig, (ax, az) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax.plot(dates, na, label=a, color='#1769aa', lw=1.1); ax.plot(dates, nb, label=b, color='#d97706', lw=1.1)
    ax.set_title(f'{a} / {b} | normalized RTH close | n={len(dates)}'); ax.set_ylabel('normalized close'); ax.grid(alpha=.25); ax.legend(frameon=False, loc='upper left')
    az.plot(dates, z, color='#374151', lw=.9); az.axhline(2, color='#b91c1c', ls='--', lw=.8); az.axhline(-2, color='#b91c1c', ls='--', lw=.8); az.axhline(0, color='#6b7280', lw=.7); az.set_ylabel('spread z'); az.grid(alpha=.25)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(); fig.tight_layout(); path = PLOTS / f'{a}_{b}.png'; fig.savefig(path, dpi=140); plt.close(fig)
    return str(path.relative_to(OUT)).replace('\\', '/')

def main():
    OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    with REVIEW.open(encoding='utf-8-sig', newline='') as h: rows = list(csv.DictReader(h))
    pool = [r for r in rows if not excluded(r)]; usage = Counter(); selected = []
    while pool and len(selected) < TARGET:
        def value(q):
            a,b=q['\u6807\u7684A'],q['\u6807\u7684B']
            return score(q)-2.6*(usage[a]+usage[b])-1.2*max(usage[a],usage[b])
        r=max(pool,key=value); selected.append(r)
        usage[r['\u6807\u7684A']]+=1; usage[r['\u6807\u7684B']]+=1; pool.remove(r)
    with DAILY.open(encoding='utf-8-sig', newline='') as h: daily=list(csv.DictReader(h))
    out=[]
    for i,r in enumerate(selected,1):
        out.append({
            'rank':i,'\u6807\u7684A':r['\u6807\u7684A'],'\u6807\u7684B':r['\u6807\u7684B'],
            '\u673a\u5236':r['\u673a\u5236'],'\u903b\u8f91\u5f3a\u5ea6':r['\u903b\u8f91\u5f3a\u5ea6'],
            '\u516c\u5171\u6837\u672c\u5929\u6570':r['\u5929\u6570'],'EG_p':r['EG_p'],'\u53cd\u5411EG_p':r['\u53cd\u5411EG_p'],
            'R2':r['R2'],'betaCV':r['betaCV'],'\u534a\u8870\u671f\u5929':r['\u534a\u8870\u671f\u5929'],'absz_gt2':r['absz'],
            '\u586b\u5145\u7387':r['\u586b\u5145\u7387'],'\u7b49\u7ea7':grade(r),
            '\u91d1\u878d\u903b\u8f91':r['\u7528\u6237\u7ed9\u7684\u7ecf\u6d4e\u903b\u8f91'],'\u9009\u62e9\u539f\u56e0':f"\u903b\u8f91\u5f3a\u5ea6={r['\u903b\u8f91\u5f3a\u5ea6']}\uff0cR2={r['R2']}\uff0cEG p={r['EG_p']}\uff0c\u534a\u8870\u671f={r['\u534a\u8870\u671f\u5929']}\u5929\uff0c\u540c\u65f6\u5bf9\u6807\u7684\u91cd\u590d\u65bd\u52a0\u60e9\u7f5a",'\u98ce\u9669\u63d0\u793a':risk(r),
            '\u4ef7\u683c\u8d70\u52bf\u56fe':plot_pair(r,daily),
            '\u6807\u7684A\u51fa\u73b0\u6b21\u6570':usage[r['\u6807\u7684A']], '\u6807\u7684B\u51fa\u73b0\u6b21\u6570':usage[r['\u6807\u7684B']]
        })
    path = OUT/'\u63a8\u8350\u4ea4\u6613\u5bf9_25\u5bf9.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    path = OUT/'\u6807\u7684\u4f7f\u7528\u6b21\u6570.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.writer(h); w.writerow(['\u6807\u7684','\u4f7f\u7528\u6b21\u6570']); w.writerows(sorted(usage.items(),key=lambda x:(-x[1],x[0])))
    with (OUT/'README.md').open('w',encoding='utf-8') as h:
        h.write('# \u63a8\u8350\u4ea4\u6613\u5bf9\uff1a\u4f4e\u6807\u7684\u91cd\u590d\n\n')
        h.write('\u57fa\u4e8e 74 \u5bf9\u5019\u9009\u8bc4\u5ba1\uff0c\u516c\u5171\u6837\u672c\u6700\u4f4e 100 \u5929\u3002\u5254\u9664 R2<0.20\u3001betaCV>2.50\u3001\u534a\u8870\u671f>400 \u5929\u3001\u586b\u5145\u7387>25% \u548c\u903b\u8f91\u5f3a\u5ea6<2 \u7684\u5019\u9009\uff0c\u5269\u4f59\u5019\u9009\u7efc\u5408\u7edf\u8ba1\u8d28\u91cf\u548c\u6807\u7684\u91cd\u590d\u60e9\u7f5a\u9009\u51fa 25 \u5bf9\u3002\n')
    print(f'selected={len(selected)} unique_symbols={len(usage)} max_usage={max(usage.values()) if usage else 0} output={OUT}')

if __name__=='__main__': main()
