"""Train a lightweight sign-crop classifier, evaluate held-out crops and export TFLite.

Data expected: --dataset ROOT containing folders 0/ ... 42/ with labeled sign
crops (e.g. GTSRB Train/). All reported accuracy is evaluated, never assumed.
A sign-crop classifier is *not* a full-scene detector.

Needs: pip install '.[train]'. Real GTSRB data is not shipped in this repository.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from edgevision.evaluation import calibration_ece


def dataset_paths(root:Path):
    files=[];classes=[]
    for folder in sorted(root.iterdir(),key=lambda x:int(x.name) if x.name.isdigit() else 999):
        if not folder.is_dir() or not folder.name.isdigit():continue
        label=int(folder.name)
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in {".ppm",".png",".jpg",".jpeg"}:
                files.append(str(p));classes.append(label)
    if not files:raise ValueError(f"No class-directory images found under {root}")
    if set(classes)!=set(range(43)):
        raise ValueError(f"Expected GTSRB classes 0..42. Found {sorted(set(classes))}")
    return np.asarray(files),np.asarray(classes,dtype=np.int32)


def split_stratified(classes:np.ndarray,val_fraction:float,seed:int):
    rng=np.random.default_rng(seed);train=[];val=[]
    for label in np.unique(classes):
        idx=np.flatnonzero(classes==label)
        rng.shuffle(idx)
        n_val=max(1,int(round(len(idx)*val_fraction)))
        if n_val>=len(idx):raise ValueError(f"Class {label} has too few samples for split")
        val.extend(idx[:n_val]);train.extend(idx[n_val:])
    return rng.permutation(train),rng.permutation(val)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True,help="GTSRB training root with class-number subdirectories")
    ap.add_argument("--out-dir",type=Path,default=Path("runs/gtsrb"))
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--image-size",type=int,default=64)
    ap.add_argument("--val-fraction",type=float,default=.20)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--quantize",action="store_true",help="Also export calibrated int8 .tflite")
    ap.add_argument("--test-csv",type=Path,help="Official held-out GTSRB Test.csv annotations (recommended)")
    ap.add_argument("--test-root",type=Path,help="Root for Test.csv relative image paths")
    args=ap.parse_args()
    if not 0<args.val_fraction<.5:raise SystemExit("--val-fraction must be in (0,.5)")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    import tensorflow as tf
    from sklearn.metrics import classification_report,confusion_matrix
    tf.keras.utils.set_random_seed(args.seed)
    x,y=dataset_paths(args.dataset)
    tri,vali=split_stratified(y,args.val_fraction,args.seed)
    size=args.image_size

    def load(path,label):
        # cv2 supports GTSRB PPM as well as PNG/JPEG; read with numpy_function
        def decode(p):
            import cv2
            img=cv2.imread(p.decode(),cv2.IMREAD_COLOR)
            if img is None:raise RuntimeError(f"Could not read {p!r}")
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        img=tf.numpy_function(decode,[path],tf.float32)
        img.set_shape((size,size,3))
        return img,label

    def dataset(index,training):
        ds=tf.data.Dataset.from_tensor_slices((x[index],y[index]))
        if training:ds=ds.shuffle(len(index),seed=args.seed,reshuffle_each_iteration=True)
        return ds.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    train=dataset(tri,True);val=dataset(vali,False)
    inputs=tf.keras.Input(shape=(size,size,3),name="rgb_sign_crop")
    h=inputs
    for filters in (24,48,96):
        h=tf.keras.layers.SeparableConv2D(filters,3,padding="same",activation="relu")(h)
        h=tf.keras.layers.BatchNormalization()(h)
        h=tf.keras.layers.MaxPooling2D()(h)
    h=tf.keras.layers.SeparableConv2D(128,3,padding="same",activation="relu")(h)
    h=tf.keras.layers.GlobalAveragePooling2D()(h)
    h=tf.keras.layers.Dropout(.25)(h)
    outputs=tf.keras.layers.Dense(43,activation="softmax",name="sign_class")(h)
    model=tf.keras.Model(inputs,outputs,name="EdgeVision_GTSRB")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=2,factor=.5)]
    history=model.fit(train,validation_data=val,epochs=args.epochs,callbacks=callbacks,verbose=2)
    test_scope="internal stratified validation split"
    evaluation=val
    labels_true=y[vali]
    if args.test_csv:
        with args.test_csv.open(encoding="utf-8-sig",newline="") as fh:
            sample=fh.read(2048);fh.seek(0)
            dialect=csv.Sniffer().sniff(sample,delimiters=",;")
            reader=csv.DictReader(fh,dialect=dialect)
            image_paths=[];official_labels=[]
            for row in reader:
                rel=row.get("Path") or row.get("Filename") or row.get("filename")
                label=row.get("ClassId") or row.get("ClassID") or row.get("class")
                if rel is None or label is None:raise ValueError("test CSV needs Path/Filename and ClassId fields")
                root=args.test_root or args.test_csv.parent
                full=root/rel
                if not full.exists():raise FileNotFoundError(full)
                image_paths.append(str(full));official_labels.append(int(label))
        test_scope="official separately labeled test CSV"
        labels_true=np.asarray(official_labels,dtype=np.int32)
        evaluation=tf.data.Dataset.from_tensor_slices((np.asarray(image_paths),labels_true)).map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size)
    probs=model.predict(evaluation,verbose=0)
    preds=np.argmax(probs,axis=1)
    report=classification_report(labels_true,preds,labels=list(range(43)),zero_division=0,output_dict=True)
    metrics={"dataset":"GTSRB user-provided labeled crops", "split":test_scope, "seed":args.seed,
             "train_samples":len(tri),"evaluated_samples":len(labels_true),"image_size":size,
             "evaluation_accuracy":float(np.mean(labels_true==preds)),
             "evaluation_macro_f1":float(report["macro avg"]["f1-score"]),
             "ece_10_bins":calibration_ece(np.max(probs,axis=1).tolist(),(labels_true==preds).astype(int).tolist()),
             "classification_report":report,"confusion_matrix":confusion_matrix(labels_true,preds,labels=list(range(43))).tolist(),
             "training_history":{k:[float(v) for v in values] for k,values in history.history.items()},
             "hardware_note":"Classifier accuracy on cropped held-out signs only; not scene detection accuracy or Raspberry Pi FPS"}
    (args.out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    model.save(str(args.out_dir/"edgevision.keras"))
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    plain=converter.convert()
    (args.out_dir/"edgevision-fp32.tflite").write_bytes(plain)
    if args.quantize:
        representative=train.unbatch().batch(1).take(100)
        quant=tf.lite.TFLiteConverter.from_keras_model(model)
        quant.optimizations=[tf.lite.Optimize.DEFAULT]
        quant.representative_dataset=lambda:([image] for image,_ in representative)
        quant.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        quant.inference_input_type=tf.int8
        quant.inference_output_type=tf.int8
        (args.out_dir/"edgevision-int8.tflite").write_bytes(quant.convert())
    labels=json.loads((Path(__file__).resolve().parents[1]/"configs/gtsrb_labels.json").read_text(encoding="utf-8"))
    (args.out_dir/"labels.json").write_text(json.dumps(labels,indent=2),encoding="utf-8")
    print(json.dumps({"evaluation_accuracy":metrics["evaluation_accuracy"],
                      "macro_f1":metrics["evaluation_macro_f1"],"ece":metrics["ece_10_bins"],
                      "artifacts":str(args.out_dir)},indent=2))

if __name__=="__main__":main()
