import time, json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, f1_score
from features import FeatureBuilder, CAT_COLS, NUM_COLS
from prep import load_clean, modelling_frame

RAW = ["created_at","cost","product_category","product_brand","product_retail_price","product_department","product_distribution_center_id"]
df = modelling_frame(load_clean())
print("rows modelling:", len(df), "sold rate:", round(df.is_sold.mean(),4))
X, y, g = df[RAW], df["is_sold"], df["product_id"]

# holdout 20% (group by product_id -> produk yang sama tidak bocor antara train/test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(X, y, g))
Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], g.iloc[tr]

def prep(kind):
    if kind == "linear":
        return ColumnTransformer([("n", StandardScaler(), NUM_COLS), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS)])
    if kind == "tree":
        return ColumnTransformer([("n","passthrough",NUM_COLS), ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS)])

def make(name):
    if name == "Logistic Regression":
        clf, k = LogisticRegression(max_iter=300, C=1.0), "linear"
    elif name == "Decision Tree":
        clf, k = DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=0), "tree"
    elif name == "Random Forest":
        clf, k = RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=100, n_jobs=1, random_state=0), "tree"
    elif name == "HistGradientBoosting":
        clf, k = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0), "tree"
    return Pipeline([("fb", FeatureBuilder()), ("pre", prep(k)), ("clf", clf)])

def best_f1_thr(y_true, p):
    ths = np.linspace(0.2, 0.8, 61); f = [f1_score(y_true, p >= t) for t in ths]
    return float(ths[int(np.argmax(f))])

res = []
gkf = GroupKFold(n_splits=3)
sub = Xtr.sample(100000, random_state=1).index
Xs, ys, gs = Xtr.loc[sub], ytr.loc[sub], gtr.loc[sub]
for name in ["Logistic Regression","Decision Tree","Random Forest","HistGradientBoosting"]:
    t=time.time(); aucs=[]
    for a,b in gkf.split(Xs, ys, gs):
        m = make(name).fit(Xs.iloc[a], ys.iloc[a]); aucs.append(roc_auc_score(ys.iloc[b], m.predict_proba(Xs.iloc[b])[:,1]))
    m = make(name).fit(Xs, ys); p = m.predict_proba(Xte)[:,1]
    r = dict(model=name, cv_auc_mean=np.mean(aucs), cv_auc_std=np.std(aucs), test_auc=roc_auc_score(yte,p),
             test_pr_auc=average_precision_score(yte,p), test_logloss=log_loss(yte,p), test_f1=f1_score(yte,p>=0.5), sec=time.time()-t)
    print(r, flush=True); res.append(r); pd.DataFrame(res).to_csv('model/baseline_results.csv', index=False)
pd.DataFrame(res).to_csv("model/baseline_results.csv", index=False)
joblib.dump((tr,te), "model/split_idx.joblib")
