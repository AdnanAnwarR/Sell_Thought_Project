import os
os.environ["TF_NUM_INTRAOP_THREADS"]="1"; os.environ["TF_NUM_INTEROP_THREADS"]="1"
import json, joblib, warnings, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from features import FeatureBuilder, CAT_COLS, NUM_COLS
from prep import load_clean, modelling_frame
tf.random.set_seed(0); np.random.seed(0)
RAW=["created_at","cost","product_category","product_brand","product_retail_price","product_department","product_distribution_center_id"]
df=modelling_frame(load_clean()); X,y=df[RAW],df.is_sold
tr,te=joblib.load("model/split_idx.joblib")
tr=np.random.RandomState(0).choice(tr,80000,replace=False)  # sampel utk hemat RAM (1 CPU, 4GB)
pre=Pipeline([("fb",FeatureBuilder()),("ct",ColumnTransformer([("n",StandardScaler(),NUM_COLS),
   ("c",OneHotEncoder(handle_unknown="ignore",min_frequency=500,sparse_output=False),CAT_COLS)]))])
Xtr=pre.fit_transform(X.iloc[tr]).astype("float32"); Xte=pre.transform(X.iloc[te]).astype("float32")
ytr,yte=y.iloc[tr].values,y.iloc[te].values; print(Xtr.shape,flush=True)
m=tf.keras.Sequential([tf.keras.layers.Input((Xtr.shape[1],)),tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(1,activation="sigmoid")])
m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
h=m.fit(Xtr,ytr,validation_split=0.15,epochs=80,batch_size=1024,verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=8,restore_best_weights=True)])
p=m.predict(Xte,batch_size=4096,verbose=0).ravel()
res={"auc":float(roc_auc_score(yte,p)),"pr_auc":float(average_precision_score(yte,p)),"logloss":float(log_loss(yte,p)),"epochs":len(h.history["loss"]),
     "params":int(m.count_params()),"input_dim":int(Xtr.shape[1])}
print(res,flush=True); json.dump({"history":{k:[float(v) for v in vs] for k,vs in h.history.items()},**res},open("model/dl_results.json","w"))
