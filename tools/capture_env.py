"""
tools/capture_env.py — à exécuter dans CHAQUE notebook (Kaggle/Colab), en fin de session d'installation.
Écrit environments/<nom>_freeze.txt (pip freeze) et <nom>_env.json (Python, OS, GPU, CUDA, frameworks).
Usage (cellule de notebook) : !python tools/capture_env.py kaggle_p100_v1    # ou colab_t4_v2, kaggle_t4_v2
"""
import json, os, platform, subprocess, sys

name = sys.argv[1]
os.makedirs("environments", exist_ok=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
open(f"environments/{name}_freeze.txt", "w").write(freeze)
info = {"python": sys.version, "platform": platform.platform()}
try:
    info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    info["nvcc"] = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
except Exception as e:
    info["gpu_error"] = str(e)
for mod in ("tensorflow", "keras", "keras_tuner", "torch", "torchvision", "ultralytics", "sklearn", "numpy", "pandas", "PIL"):
    try:
        m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        if mod == "torch": info["torch_cuda"] = m.version.cuda
    except Exception:
        info[mod] = None
json.dump(info, open(f"environments/{name}_env.json", "w"), indent=2)
print(json.dumps(info, indent=2))
