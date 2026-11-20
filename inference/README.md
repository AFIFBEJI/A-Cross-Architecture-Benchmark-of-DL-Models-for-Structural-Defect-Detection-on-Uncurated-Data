# inference/

`predict_folder.py` : inférence **sans entraînement** d'un checkpoint existant (`.keras` ou `.pt`) sur un dossier `classe/image`. Produit `predictions_<modèle>.csv` (path, true, pred, p_<classe>), entrée des scripts de `../evaluation/`.

> Les poids ne sont pas versionnés (voir `../README.md`, section *Data and Model Weights*). Les modèles v1 .tflite du dépôt ne sont pas chargeables par ce script : utiliser les `.keras` / `.pt` du Drive.
>
> Le protocole **zero-shot** décrit dans le README racine (sans fine-tuning, 21 013 images non curées) n'a **pas** été exécuté : aucun script de ce type n'existe car aucun résultat n'en provient. Voir `../EXPERIMENT_INDEX.md`.
