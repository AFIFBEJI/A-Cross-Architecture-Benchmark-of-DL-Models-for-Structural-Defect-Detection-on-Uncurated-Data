"""
code/v2/train_pytorch_reference.py

!! RECONSTRUCTION DE RÉFÉRENCE — PAS le notebook d'origine !!
Les notebooks Colab/Kaggle réellement exécutés pour la v2 (ResNet50, EfficientNet-B0, EfficientNet-B2) ne sont
pas dans ce dépôt. Ce script réimplémente le protocole décrit par
models/<modèle>/v2/Hyperparametres/hyperparametres_*.json et rapport_*.md (PyTorch/torchvision, AdamW,
CosineAnnealingLR, label smoothing 0.1, early stopping patience 7, seed 42, 30 epochs max).
Il ne prouve pas l'origine des fichiers de models/*/v2/Resultats/ : seul l'export .ipynb original (avec sorties) le fera.

Usage : python code/v2/train_pytorch_reference.py --arch efficientnet_b0 --data /chemin/dataset   # dataset/{train,val}/<classe>
"""
import argparse, copy, random
import numpy as np, torch, torch.nn as nn
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader

CFG = {  # valeurs = hyperparametres_*.json
    "resnet50": dict(size=224, batch=32, dropout=None),
    "efficientnet_b0": dict(size=224, batch=32, dropout=0.2),
    "efficientnet_b2": dict(size=260, batch=24, dropout=0.3),
}
ap = argparse.ArgumentParser(); ap.add_argument("--arch", choices=CFG, required=True); ap.add_argument("--data", required=True)
a = ap.parse_args(); c = CFG[a.arch]
random.seed(42); np.random.seed(42); torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr = T.Compose([T.RandomResizedCrop(c["size"]), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
te = T.Compose([T.Resize((c["size"], c["size"])), T.ToTensor(), norm])
dtr, dva = datasets.ImageFolder(f"{a.data}/train", tr), datasets.ImageFolder(f"{a.data}/val", te)
ltr, lva = DataLoader(dtr, c["batch"], shuffle=True, num_workers=2), DataLoader(dva, c["batch"], num_workers=2)
if a.arch == "resnet50":
    m = models.resnet50(weights="IMAGENET1K_V2"); m.fc = nn.Linear(m.fc.in_features, 4)
else:
    m = getattr(models, a.arch)(weights="IMAGENET1K_V1")
    m.classifier = nn.Sequential(nn.Dropout(c["dropout"]), nn.Linear(m.classifier[1].in_features, 4))
m.to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
best, bad, best_w = 0, 0, None
for ep in range(30):
    m.train()
    for x, y in ltr:
        opt.zero_grad(); crit(m(x.to(dev)), y.to(dev)).backward(); opt.step()
    sch.step(); m.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in lva:
            ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    acc = ok / n; print(f"epoch {ep+1} val_acc={acc:.4f}")
    if acc > best: best, bad, best_w = acc, 0, copy.deepcopy(m.state_dict())
    else:
        bad += 1
        if bad >= 7: break
torch.save(best_w, f"{a.arch}_v2_reference.pth")
