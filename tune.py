import json, joblib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from features import FeatureBuilder, CAT_COLS, NUM_COLS, CENSOR_CUTOFF
from prep import load_clean, modelling_frame
RAW = ["created_at","cost","product_category","product_brand","product_retail_price","product_department","product_distribution_center_id"]
df = modelling_frame(load_clean()); X,y,g = df[RAW], df.is_sold, df.product_id
tr,te = joblib.load("model/split_idx.joblib")
Xtr,Xte,ytr,yte,gtr = X.iloc[tr],X.iloc[te],y.iloc[tr],y.iloc[te],g.iloc[tr]
out = {}
def pipe(cols_num=NUM_COLS, cols_cat=CAT_COLS, **kw):
    pre = ColumnTransformer([("n","passthrough",cols_num),("c",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cols_cat)])
    return Pipeline([("fb",FeatureBuilder()),("pre",pre),("clf",HistGradientBoostingClassifier(random_state=0, **kw))])

# 1) Ablation: fitur mana yang benar-benar punya sinyal?
abl = {}
sets = {"Hanya waktu (t_days)": (["t_days"],[]),
        "Waktu + musim": (["t_days","month_sin","month_cos","dow"],[]),
        "Tanpa waktu (harga/kategori/brand/DC)": (["log_price","log_cost","margin_pct","margin_abs"],CAT_COLS),
        "Semua fitur": (NUM_COLS,CAT_COLS)}
for k,(n,c) in sets.items():
    m = pipe(n,c,max_iter=150,learning_rate=0.08).fit(Xtr,ytr); abl[k]=float(roc_auc_score(yte,m.predict_proba(Xte)[:,1]))
    print("ablation",k,abl[k],flush=True)
out["ablation_auc"]=abl

# 2) Uji ekstrapolasi temporal: train < 2023-06, test 2023-06..cutoff
tm = df.created_at < "2023-06-01"; Xa,ya = X[tm],y[tm]; Xb,yb = X[~tm],y[~tm]
m = pipe(max_iter=150,learning_rate=0.08).fit(Xa,ya); out["temporal_holdout_auc"]=float(roc_auc_score(yb,m.predict_proba(Xb)[:,1]))
out["temporal_holdout_base_rate_train"]=float(ya.mean()); out["temporal_holdout_base_rate_test"]=float(yb.mean())
print("temporal",out["temporal_holdout_auc"],flush=True)

# 3) Hyperparameter tuning HGB (RandomizedSearch, GroupKFold)
sub = Xtr.sample(120000,random_state=2).index
params = {"clf__learning_rate":[0.03,0.05,0.08,0.12],"clf__max_iter":[150,250,350],"clf__max_leaf_nodes":[15,31,63],
          "clf__min_samples_leaf":[20,100,300],"clf__l2_regularization":[0,1,10],"clf__max_depth":[None,6,10]}
rs = RandomizedSearchCV(pipe(), params, n_iter=10, scoring="roc_auc", cv=GroupKFold(3), random_state=0, n_jobs=1, refit=False)
rs.fit(Xtr.loc[sub], ytr.loc[sub], groups=gtr.loc[sub])
best = {k.replace("clf__",""):v for k,v in rs.best_params_.items()}
cvr = pd.DataFrame(rs.cv_results_)[["params","mean_test_score","std_test_score"]].sort_values("mean_test_score",ascending=False)
cvr.to_csv("model/tuning_results.csv",index=False); out["best_params"]={k:(None if v is None else float(v) if isinstance(v,float) else int(v)) for k,v in best.items()}
out["tuning_best_cv_auc"]=float(rs.best_score_); print("best",best,rs.best_score_,flush=True)

# 4) Model akhir: fit pada train, evaluasi holdout
final = pipe(**best).fit(Xtr,ytr); p = final.predict_proba(Xte)[:,1]
ths=np.linspace(0.2,0.7,51); f1s=[f1_score(yte,p>=t) for t in ths]; thr=float(ths[int(np.argmax(f1s))])
out["final_holdout"]={"auc":float(roc_auc_score(yte,p)),"pr_auc":float(average_precision_score(yte,p)),"logloss":float(log_loss(yte,p)),
  "brier":float(brier_score_loss(yte,p)),"threshold":thr,"f1":float(f1_score(yte,p>=thr)),"precision":float(precision_score(yte,p>=thr)),
  "recall":float(recall_score(yte,p>=thr)),"base_rate":float(yte.mean())}
print(out["final_holdout"],flush=True)
np.save("model/holdout_proba.npy",p); np.save("model/holdout_y.npy",yte.values)
# calibration by decile
q = pd.qcut(p,10,duplicates="drop"); cal = pd.DataFrame({"p":p,"y":yte.values}).groupby(q).mean(); cal.to_csv("model/calibration.csv",index=False)
# permutation importance (grup fitur mentah)
idx = Xte.sample(15000,random_state=0).index; Xi,yi = Xte.loc[idx],yte.loc[idx]
base = roc_auc_score(yi,final.predict_proba(Xi)[:,1]); imp={}
rng=np.random.RandomState(0)
for col in RAW:
    Xp=Xi.copy(); Xp[col]=rng.permutation(Xp[col].values); imp[col]=float(base-roc_auc_score(yi,final.predict_proba(Xp)[:,1]))
imp["cost+price (margin)"]=None; imp.pop("cost+price (margin)")
out["perm_importance_auc_drop"]=imp; print(imp,flush=True)

# 5) Refit pada seluruh data model -> artefak deployment
deploy = pipe(**best).fit(X,y)
joblib.dump(deploy,"model/model.joblib",compress=3)
meta = {"threshold":thr,"base_rate":float(y.mean()),"train_rows":int(len(X)),"cutoff":str(CENSOR_CUTOFF.date()),
  "categories":sorted(df.product_category.unique().tolist()),"departments":sorted(df.product_department.unique().tolist()),
  "dcs":sorted(df.product_distribution_center_id.unique().tolist()),
  "top_brands":df.product_brand.value_counts().head(60).index.tolist(),
  "price_range":[float(df.product_retail_price.min()),float(df.product_retail_price.max())],
  "avg_cost_ratio_by_cat":(df.assign(r=df.cost/df.product_retail_price).groupby("product_category").r.mean().round(3).to_dict()),
  "last_stock_date":str(df.created_at.max().date())}
json.dump(meta,open("model/meta.json","w"),indent=1); json.dump(out,open("model/results.json","w"),indent=1)
print("DONE",flush=True)
