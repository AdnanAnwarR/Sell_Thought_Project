import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve
from prep import load_clean, modelling_frame
sns.set_theme(style="whitegrid", font_scale=1.0); C="#1f6f8b"; C2="#e07a5f"
df=load_clean(); m=modelling_frame(df)
# 1 sold rate by creation quarter (+ censoring)
q=df.groupby(df.created_at.dt.tz_localize(None).dt.to_period("Q").astype(str)).is_sold.agg(["size","mean"]).iloc[2:]
fig,ax=plt.subplots(figsize=(8,4)); ax.bar(q.index,q["mean"]*100,color=C); ax.set_ylabel("% item terjual"); ax.set_title("Persentase item terjual per kuartal pembuatan stok")
plt.xticks(rotation=60); fig.tight_layout(); fig.savefig("figures/f1_sold_by_quarter.png",dpi=160); plt.close()
# 2 days to sell
d=(df.sold_at-df.created_at).dt.total_seconds()/86400
fig,ax=plt.subplots(figsize=(6,4)); ax.hist(d.dropna(),bins=60,color=C); ax.set_xlabel("Hari dari stok dibuat sampai terjual"); ax.set_title("Distribusi waktu terjual (seragam 0-60 hari)")
fig.tight_layout(); fig.savefig("figures/f2_days_to_sell.png",dpi=160); plt.close()
# 3 sold rate by category (flat)
c=m.groupby("product_category").is_sold.mean().sort_values()*100
fig,ax=plt.subplots(figsize=(7,6)); ax.barh(c.index,c.values,color=C); ax.axvline(m.is_sold.mean()*100,color=C2,ls="--"); ax.set_xlim(0,60); ax.set_xlabel("% terjual"); ax.set_title("Sell-through per kategori: hampir identik (~36%)")
fig.tight_layout(); fig.savefig("figures/f3_sold_by_category.png",dpi=160); plt.close()
# 4 margin
mm=m.assign(mp=(m.product_retail_price-m.cost)/m.product_retail_price)
fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].hist(mm.mp,bins=50,color=C); ax[0].set_title("Distribusi margin (%)")
sns.boxplot(data=mm,x="is_sold",y="product_retail_price",ax=ax[1],color=C,showfliers=False); ax[1].set_title("Harga jual vs status terjual"); ax[1].set_xticklabels(["Tidak terjual","Terjual"]); ax[1].set_xlabel("")
fig.tight_layout(); fig.savefig("figures/f4_margin_price.png",dpi=160); plt.close()
# 5 monthly sold rate trend
mo=m.groupby(m.created_at.dt.tz_localize(None).dt.to_period("M").astype(str)).is_sold.mean().iloc[14:]
fig,ax=plt.subplots(figsize=(8,4)); ax.plot(mo.index,mo.values*100,color=C2,lw=2); ax.set_xticks(range(0,len(mo),4)); ax.set_xticklabels(mo.index[::4],rotation=45); ax.set_ylabel("% terjual"); ax.set_title("Tren sell-through bulanan (setelah handling censoring)")
fig.tight_layout(); fig.savefig("figures/f5_monthly_trend.png",dpi=160); plt.close()
# 6 ROC + ablation + importance
r=json.load(open("model/results.json")); p=np.load("model/holdout_proba.npy"); y=np.load("model/holdout_y.npy")
fpr,tpr,_=roc_curve(y,p); fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].plot(fpr,tpr,color=C,lw=2,label=f"HGB AUC={r['final_holdout']['auc']:.3f}"); ax[0].plot([0,1],[0,1],"k--",label="Acak"); ax[0].legend(); ax[0].set_title("ROC Curve (holdout)")
a=r["ablation_auc"]; ax[1].barh(list(a.keys()),list(a.values()),color=[C,C,C2,C]); ax[1].axvline(.5,color="k",ls="--"); ax[1].set_xlim(.45,.72); ax[1].set_title("Ablasi fitur (AUC)"); 
for i,v in enumerate(a.values()): ax[1].text(v+.002,i,f"{v:.3f}",va="center")
pi=r["perm_importance_auc_drop"]; ax[2].barh(list(pi.keys()),list(pi.values()),color=C); ax[2].set_title("Permutation importance (penurunan AUC)")
fig.tight_layout(); fig.savefig("figures/f6_model_eval.png",dpi=160); plt.close()
# 7 calibration
cal=pd.read_csv("model/calibration.csv"); fig,ax=plt.subplots(figsize=(4.8,4.4)); ax.plot(cal.p,cal.y,"o-",color=C); ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("Probabilitas prediksi"); ax.set_ylabel("Proporsi terjual aktual"); ax.set_title("Kalibrasi (decile)")
fig.tight_layout(); fig.savefig("figures/f7_calibration.png",dpi=160); plt.close()
print("figs ok")
