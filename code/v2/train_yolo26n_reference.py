"""
RECONSTRUCTION DE RÉFÉRENCE (pas le notebook d'origine) — YOLO26n-cls v2.
Paramètres = models/yolov26_cls/v2/Hyperparametres/hyperparametres_yolo26.json.
Usage : python code/v2/train_yolo26n_reference.py /chemin/dataset   # dataset/{train,val}/<classe>
"""
import sys
from ultralytics import YOLO

YOLO("yolo26n-cls.pt").train(data=sys.argv[1], epochs=100, imgsz=224, batch=64, optimizer="auto", lr0=0.001, lrf=0.01,
                             momentum=0.937, weight_decay=0.0005, warmup_epochs=3, patience=20, dropout=0.1,
                             augment=True, device=0, seed=42, pretrained=True, val=True,
                             project="runs/classify", name="YOLO26_2")
