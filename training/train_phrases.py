from pathlib import Path
import json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

ROOT=Path("datasets/phrases"); MODEL=Path("models"); MODEL.mkdir(exist_ok=True)
X=[]; y=[]
for d in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    for f in d.glob("*.npy"):
        a=np.load(f)
        if a.shape==(30,63): X.append(a); y.append(d.name.replace("_"," "))
if len(set(y))<2: raise SystemExit("Collect at least 2 phrase classes first.")
X=np.asarray(X,dtype="float32"); enc=LabelEncoder(); yi=enc.fit_transform(y)
Xtr,Xte,ytr,yte=train_test_split(X,yi,test_size=.2,random_state=42,stratify=yi)
m=keras.Sequential([keras.layers.Input((30,63)),keras.layers.GRU(128,return_sequences=True),
 keras.layers.Dropout(.25),keras.layers.GRU(64),keras.layers.Dense(64,activation="relu"),
 keras.layers.Dense(len(enc.classes_),activation="softmax")])
m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
m.fit(Xtr,ytr,validation_split=.15,epochs=70,batch_size=16,
 callbacks=[keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)],verbose=2)
print("Test:",m.evaluate(Xte,yte,verbose=0))
m.save(MODEL/"phrase_sequence.keras")
(MODEL/"phrase_labels.json").write_text(json.dumps(enc.classes_.tolist(),indent=2),encoding="utf-8")
