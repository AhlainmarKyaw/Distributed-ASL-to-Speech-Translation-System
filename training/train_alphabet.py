from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow import keras

DATA=Path("datasets/alphabet_landmarks.csv"); MODELS=Path("models"); MODELS.mkdir(exist_ok=True)
if not DATA.exists(): raise SystemExit("Missing datasets/alphabet_landmarks.csv. Run training/extract_alphabet_landmarks.py first.")
df=pd.read_csv(DATA); X=df.drop(columns=["label"]).values.astype("float32"); y=df["label"].astype(str).values
enc=LabelEncoder(); yi=enc.fit_transform(y)
Xtr,Xte,ytr,yte=train_test_split(X,yi,test_size=.2,random_state=42,stratify=yi)
model=keras.Sequential([
 keras.layers.Input((63,)), keras.layers.Dense(256,activation="relu"), keras.layers.BatchNormalization(),
 keras.layers.Dropout(.25), keras.layers.Dense(128,activation="relu"), keras.layers.Dropout(.2),
 keras.layers.Dense(len(enc.classes_),activation="softmax")
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
cb=[keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(patience=4)]
model.fit(Xtr,ytr,validation_split=.15,epochs=60,batch_size=64,callbacks=cb,verbose=2)
loss,acc=model.evaluate(Xte,yte,verbose=0); print(f"Test accuracy: {acc:.4f}")
pred=model.predict(Xte,verbose=0).argmax(1); print(classification_report(yte,pred,target_names=enc.classes_,zero_division=0))
model.save(MODELS/"alphabet_landmarks.keras")
(MODELS/"alphabet_labels.json").write_text(json.dumps(enc.classes_.tolist(),indent=2),encoding="utf-8")
print("Saved models/alphabet_landmarks.keras")
