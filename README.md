# DeepCBIR

**Content-Based Image Retrieval with pretrained CNN descriptors** — a reproducible benchmark of lightweight and
classic backbones on Corel (1K/5K/10K), CIFAR-100 and Paris6k.

> Master 2 SID · USTO-MB · module IRDM (Information Retrieval & Data Mining).
> The generated report, web app and static site are in **French**.

## Goal
Given a query image, retrieve the most similar images of a database using only deep features
(no fine-tuning), then measure how good the retrieval is. The project compares:

- **5 networks** (ImageNet-pretrained): SqueezeNet 1.1 (512-d), MobileNetV3-Small (576-d), ResNet-18 (512-d),
  ResNet-50 (2048-d), VGG-16 (4096-d, fc7)
- **2 similarities**: cosine vs Euclidean
- **5 databases**: Corel-1K (1,000 img / 10 classes), Corel-5K (5,000 / 50), Corel-10K (10,000 / 100),
  CIFAR-100 test (10,000 / 100, 32×32 PNG), Paris6k (6,412 / 11 landmarks + distractors)
- **Metrics**: Precision-Recall curve, AP / mAP, P@K, R@K — globally, **per class**, per database and per file type

## Pipeline
| Step | Script | What it does |
|---|---|---|
| 0 | `etape0_planification.py` | Benchmarks *your* machine and proposes a plan that fits a time budget |
| 1 | `etape1_preparation.py` | Scans databases, 224×224 resize + ImageNet normalisation, coverage report |
| 2 | `etape2_extraction.py` | Offline indexing: one feature vector per image (resumable, checkpoints every ~1000 images) |
| 3 | `etape3_recherche.py` | Online retrieval: Top-K for a query, cosine vs Euclidean |
| 4 | `etape4_evaluation.py` | PR curves, mAP, P@K, R@K, per-class AP, confusion matrices (incremental) |
| 5 | `etape5_rapport.py` | Full report (Markdown + standalone HTML) |
| 6 | `etape6_site_html.py` | Multi-page offline HTML site with all results and explanations |
| – | `app.py` | Interactive Streamlit web app (search, results, comparisons) |

## Quick start
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python etape0_planification.py --budget-min 90   # pick a plan for your machine
python etape1_preparation.py
python etape2_extraction.py --models squeezenet1_1 mobilenet_v3_small resnet18 resnet50
python etape4_evaluation.py
python etape6_site_html.py                       # open results/site/index.html
streamlit run app.py
```

**Data layout** — datasets are *not* in the repository. Put them next to the repo folder
(or set the `CBIR_DATA_DIR` environment variable):
```
MiniProject/
├── CIFAR100/TEST/<class>/*.png
├── Corel-1K/ Corel-5K/ Corel-10K/        (<class>/*.jpg|jpeg)
├── paris/paris_1/ paris/paris_2/         (paris_<class>_NNNNNN.jpg)
└── deep-cbir/                            <- this repository
```

## Speed & reproducibility
- Each image is decoded once for all models; `channels_last`, all CPU cores, automatic GPU.
- Interrupt anytime (or use `--time-limit-min`) and rerun the same command: it resumes.
- Evaluation is incremental; fixed random seed; leave-one-out protocol (a query never retrieves itself).
- Paris6k: the `general` distractor class stays in the database but is never used as a query.

## Notes & limitations
Features are used as-is (no fine-tuning); relevance means *same class*; CIFAR-100 images are upscaled from 32 to 224 px.
Results are generated locally into `results/` (not versioned). Pretrained weights are downloaded by torchvision.

## License
MIT (add a `LICENSE` file when creating the repository on GitHub).