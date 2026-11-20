# code/v2 — statut

La v2 a été exécutée dans des notebooks Colab (ResNet50, EfficientNet-B0, Drive) et Kaggle (EfficientNet-B2, YOLO26n) dont l'export `.ipynb` **n'est pas encore dans le dépôt**. Les deux scripts présents sont des reconstructions fidèles aux JSON/rapports v2, fournies pour documenter le protocole, pas pour prouver l'origine des résultats.

**À faire (auteur de la v2)** : exporter chaque notebook (*File → Download → .ipynb*, sorties conservées) vers `notebooks/v2/` sous le nom `v2_<modèle>_<plateforme>.ipynb`, puis mettre à jour la colonne « Code » de `EXPERIMENT_INDEX.md`.

Anomalies v2 relevées pendant l'audit (détail dans `EXPERIMENT_INDEX.md`) : ResNet50 et EfficientNet-B0 ont été testés sur un `val/` ne contenant que `corrosion` (accuracy 99.17 % / 97.01 % non représentatives) ; le dataset n'est pas identique entre modèles ; YOLO `results.csv` s'arrête à 22 epochs alors que le rapport mentionne l'arrêt à l'epoch 64.
