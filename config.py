"""Configuration centrale du mini-projet CBIR (IRDM - Master 2 SID, USTO-MB)."""
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent
# Structure : MiniProject/{CIFAR100,Corel-1K,Corel-5K,Corel-10K,paris,script/}
# -> le code est dans script/, les bases sont dans le dossier PARENT.
DATA_DIR = Path(os.environ.get("CBIR_DATA_DIR", BASE.parent))   # variable d'environnement facultative
WORK = BASE / "workdir"
MANIFESTS = WORK / "manifests"
FEATURES = WORK / "features"
RESULTS = BASE / "results"
FIGS = RESULTS / "figures"
CONF_DIR = RESULTS / "confusion"
for d in (WORK, MANIFESTS, FEATURES, RESULTS, FIGS, CONF_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# Jeux de données : dossier(s) possible(s) sous data/, extensions, règle de classe
# ----------------------------------------------------------------------------
DATASETS = {
    "corel1k": dict(
        label="Corel-1K", roots=["Corel-1K"], exts=(".jpg", ".jpeg"),
    ),
    "cifar100": dict(
        label="CIFAR-100", roots=["CIFAR100"], find_dir="test",   # partition TEST (10 000 images) : dossier TEST trouvé automatiquement
        exts=(".png",),
    ),
    "paris6k": dict(
        label="Paris6k", roots=["paris"], subdirs=["paris_1", "paris_2"],  # oxbuild_images (Oxford) ignoré
        exts=(".jpg", ".jpeg"),
        class_regex=r"paris_([a-z]+)_\d+",   # ex: paris_eiffel_000123.jpg -> eiffel
        strict_regex=True,                    # fichiers ne respectant pas ce nom : ignorés
        query_exclude=["general"],            # distracteurs : jamais utilisés comme requêtes
    ),
    "corel5k": dict(label="Corel-5K", roots=["Corel-5K"], exts=(".jpg", ".jpeg")),     # 50 classes
    "corel10k": dict(label="Corel-10K", roots=["Corel-10K"], exts=(".jpg", ".jpeg")),  # 100 classes
    # OPTION lourde (60 000 images TRAIN+TEST+VALID) : --datasets cifar100_all  (voir README)
    "cifar100_all": dict(label="CIFAR-100 (60 000)", roots=["CIFAR100"], exts=(".png",)),
}
DEFAULT_DATASETS = ["corel1k", "corel5k", "corel10k", "cifar100", "paris6k"]   # toutes les bases

# Nombre max d'images par classe (None = TOUTES). "full" = couverture 100 % (défaut) ; "quick" = test rapide.
PRESETS = {
    "quick": {"corel1k": None, "corel5k": 40, "corel10k": 20, "cifar100": 20, "paris6k": 80, "cifar100_all": 20},
    "full": {k: None for k in ("corel1k", "corel5k", "corel10k", "cifar100", "paris6k", "cifar100_all")},
}

MODELS = {   # clé -> (nom affiché, famille)
    "squeezenet1_1": ("SqueezeNet 1.1", "Léger"),
    "mobilenet_v3_small": ("MobileNetV3-Small", "Léger"),
    "resnet18": ("ResNet-18", "Classique"),
    "resnet50": ("ResNet-50", "Classique"),
    "vgg16": ("VGG-16", "Classique"),
}
METRICS = ["cosinus", "euclidienne"]
K_LIST = [1, 5, 10, 20, 50]
RECALL_LEVELS = [i / 10 for i in range(11)]
