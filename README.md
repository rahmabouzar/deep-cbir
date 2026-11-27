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
| 6 | `etape6_site_html.py` | Offline multi-page HTML site: **one page per stage** (inputs, outputs, results, how to read them, downloadable result images) + comparisons and conclusion. Also runs step 7 |
| 7 | `etape7_pdf.py` | Result images (PNG) for every stage + **complete PDF report** (clickable table of contents) and one PDF per stage |
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
Example results are generated locally into `results/`. Pretrained weights are downloaded by torchvision.

## License
MIT (add a `LICENSE` file when creating the repository on GitHub).

## 🎲 Random-sample experiment (new)
Web page **« 🎲 Expérience aléatoire »** in `streamlit run app.py` (backend: `experience.py`, page: `page_experience.py`).
Pick a database → *Tirer l'échantillon* (random classes + random images, preview, seed shown) → *Lancer l'évaluation*
(extraction + cosine/Euclidean retrieval + metrics run in Python, progress bar) → table: rows = model × {P@10 moyen, mAP max, mAP moyen},
columns = the drawn classes (+ « Toutes classes »). Corel-1K: its 10 classes; Paris6k: all landmarks; Corel-5K: 10; Corel-10K / CIFAR-100: 20 (10 or 20 selectable).
Command line: `python experience.py --dataset corel10k --classes 20 --seed 7` (CSV saved in `results/experiences/`).


## Web interface (updated)
`python -m streamlit run app.py` — grouped sidebar navigation with Material icons (no emojis): Accueil · Expérience sur échantillon · Recherche d'images ·
Résultats globaux · Courbes Précision-Rappel · Comparaison par classe · Matrices de confusion · Comparaisons croisées.
The in-app « Rapport » page was removed (`etape5_rapport.py` still generates `results/rapport.html` from the command line).
Theme: `.streamlit/config.toml` (colors, hidden Deploy toolbar) + `ui.py` (CSS and Plotly style shared by all pages).
Corel-5K/10K classes are named « 0 », « 1 », … : heatmaps now use categorical axes so class order and labels display correctly.

## Design system
Rules come from `SKILL.md` (landing-page-design, part B: Geist font, Tailwind type scale with named text styles, spacing tokens, Tailwind radii,
no one-sided borders, one motion curve, no Material icons) and the palette from `DESIGN-notion.md`. Everything is defined once in `ui.py`.
Run `python design_check.py` after any style change: it verifies the generated CSS and the Python sources against those rules (exit code 1 on any violation).
Icons: `assets/icons/<name>.svg` (your own exports, e.g. Streamline) > `assets/phosphor/<name>.svg` (Phosphor, MIT) > built-in fallback.

## Search page: precision and evaluation tools
« Recherche d'images » now shows P@K, R@K, AP (full ranking), R-precision, rank of the first relevant image, per-query precision-recall curves,
the class predicted by K-neighbour vote, a cross-model consensus (Jaccard agreement + RRF fusion) and a robustness test (rotation, crop, blur, noise, JPEG...).
Logic lives in `recherche_eval.py`. For an uploaded image, pick the expected class to get precision.

## Query image selection (Recherche d'images similaires)
Three ways to choose the query: **Boîte de dialogue** (native file window opened by `dialogue.py`, tkinter in a separate process),
**Parcourir la base** (class + thumbnails + « Choisir »), or **Téléverser dans le navigateur**.
An image that already belongs to the base is detected (by path, or by identical descriptor): it is excluded from its own results and its class is known.
For any other image, pick the expected class to get precision. The experience page keeps both modes: random draw, and « Je choisis les classes et les images » (class checkboxes, thumbnails, all / none / n random per class).

## Colors and results labels
Each page has its own main color from the sticker palette (tinted background, buttons, steps, tabs, active menu entry, first chart color): `ui.accent_css`.
In the search page, results similar to the query class are written in green (« Similaire ») and the others in red (« Différente »).
Choosing a file from another base (e.g. a Paris6k image while Corel-1K is selected) switches the base automatically.
Without an expected class, the reference is the class predicted by vote, and the page says so.

## Pages Projet et Étapes (new)
Sidebar groups **Le projet** (Architecture, Technologies, Formules et calculs) and **Étapes du pipeline** (one results page per script: étape 0 to 4, and 5 to 7).
Code: `pages_projet.py`, `pages_etapes.py`, helpers in `pages_outils.py`; animated components (pipeline, tree, cards, bars, rank strip) live in `ui.py`.
Each stage page shows inputs → script → outputs, the real results read from `workdir/` and `results/` (or the command to run if absent) and the real source of the key function.
Animations use opacity/transform only and stop with `prefers-reduced-motion`. `python design_check.py` also covers the new files.
The dataset label is now « CIFAR-100 » (partition TEST); results saved with the old label are relabelled on read.
