VenomAI

Explainable machine learning for predicting venom peptide targets (and testing whether toxicity can be predicted).

VenomAI takes the amino acid sequence of a spider, scorpion or cone snail venom peptide and predicts which kind of molecular target it most likely acts on (for example a sodium channel or a potassium channel). It then shows, in plain language, why the model answered the way it did. It also reports honestly where it is weak.

Research prototype: Predictions are statistical guesses from public database annotations, not laboratory results. They must be validated experimentally before any biological or therapeutic claim is made. Not for clinical or medical use.

# Contents
What it does
Key results (v3.2)
How it works
Repository structure
Quick start: run the website
Retrain the model in Google Colab
Using the predictor from Python
Data sources and labels
Limitations
Troubleshooting
Citation
License

# What it does

Target prediction. Multiclass classification into eight target classes: voltage-gated sodium channel, potassium channel, calcium channel, glutamate receptor, nicotinic acetylcholine receptor, G-protein-coupled receptor, enzyme, and membrane-associated target.
Plain-language results. The website shows the top prediction, a confidence wording, the probability of each class, and a note on how reliable the model is for the chosen animal group.
Explanations. A "why did the model say this" page shows which sequence features pushed the answer, using SHAP, with plain-English feature names.
Toxicity, tested honestly. A toxicity model was trained and tested. It did not pass our reliability checks, so the website deliberately shows no toxicity verdict. See Key results.

# Key results (v3.2)

All numbers use a leakage-aware split: sequences at least 70% identical (CD-HIT) always stay on the same side of the train/test split, so the model cannot score well just by recognising near-copies.

Target classification (1,024 labelled peptides; 833 train / 191 test)

Model	Accuracy	Balanced accuracy	Macro F1
XGBoost	0.864	0.782	0.820
Random Forest (used by the website)	0.770	0.735	0.737
Neural network	0.775	0.664	0.692
SVM	0.743	0.666	0.692

The website uses Random Forest because the saved XGBoost model could not be loaded reliably in the deployment environment (see Troubleshooting). Some classes have only 4 to 8 test peptides, so per-class scores are noisy.

Within each animal group (untuned Random Forest, balanced accuracy)

Group	Score	Test peptides	Evidence level
Spider	about 0.70	42	Preliminary
Cone snail	about 0.68	68	Preliminary
Scorpion	about 0.64	88	Preliminary

These are lower than the overall score, which means part of the overall result reflects which animal a peptide came from, not only what it targets. The scores also move when the data is split differently, so read them as approximate.

Toxicity (273 labelled peptides: 208 toxic, 65 non-toxic): not reliable.

Test	Model	Length + cysteine baseline
Full set, repeated cluster-based cross-validation	0.673	0.699
Matched (hard) set, same method	0.546	0.506
Full set, with ESM-2 embeddings	0.706	0.699
Matched set, with ESM-2 embeddings	0.559	0.506

No model beat the simple length-and-cysteine baseline by a margin larger than its own spread, so the reliability gate fails and the site withholds toxicity results. The most likely cause is the labels: only 65 non-toxic examples, and they are proteins without a toxin tag, not experimentally confirmed non-toxic peptides. We treat this as a tested negative result, not something to tune away. The reliability thresholds were not loosened to make it pass.

# How it works

Collect peptide sequences from ArachnoServer, ConoServer and UniProt, plus non-venom control proteins from the same taxa.
Clean and label. Remove duplicates and invalid sequences. Target labels come from text rules over database annotations. For ConoServer, only mature (wild type) peptides are kept, and a documented table of curated class names (for example conantokin to glutamate receptor) fills labels only where the text rules found nothing.
Featurize. Amino acid composition, dipeptide composition, physicochemical properties, cysteine spacing, and 84 CTD (Composition/Transition/Distribution) descriptors. An optional ESM-2 embedding hook is off by default.
Split fairly. CD-HIT clusters at 70% identity, then a grouped train/test split.
Train Random Forest, SVM, XGBoost and a neural network, tuned with leakage-aware randomized search. Probabilities are calibrated.
Evaluate with accuracy, balanced accuracy, macro and weighted F1, per-class metrics, baselines, a within-animal check, and a toxicity reliability gate.
Explain with SHAP (exact TreeExplainer for tree models, sampled explainer for SVM and neural network).
Serve through a small Flask website.

The training script is organised in numbered sections:

Section	Purpose
0 to 3	Install/import, configuration, HTTP helper, sequence utilities
4	Data sources (ArachnoServer, ConoServer, UniProt)
5	Labelling (target and toxicity rules, control cleaning)
6	CD-HIT clustering for leakage control
7	Feature engineering
8	Models, tuning, shared train/evaluate loop
9	SHAP explanations
10	UniProt pathway cross-references (a lookup, not a prediction)
11	Predicting and ranking new peptides
12 to 13	Pipeline orchestration and the run command

# Repository structure

Adjust this tree to match your repository.

VenomAI/
├── app.py                      # Flask app: pages and /predict, /explain
├── venomai_v3.py               # Predictor and feature code imported by the website
├── test_load.py                # Checks that the model file loads and predicts
├── requirements-web.txt        # Libraries for running the website
├── templates/
│   ├── index.html              # Landing page
│   ├── tool.html               # Paste a sequence, get results
│   └── why.html                # "Why did the model say this" page
├── static/
│   ├── style.css
│   ├── app.js                  # Tool page logic
│   ├── why.js                  # Explanation page logic
│   └── images/hero.jpg
├── model/
│   └── venomai_predictor_web.joblib   # NOT in the repository: you create it (see below)
└── colab/
    └── VenomAI_Colab_v3_2.py   # Full training pipeline (run in Google Colab)

The trained model is not stored in this repository. It is about 44 MB and is tied to specific library versions, so you generate it yourself by running the training script in Google Colab (see Retrain the model in Google Colab) and saving the lighter web copy into model/. The website will not start until model/venomai_predictor_web.joblib exists.

# Quick start: run the website

Requires Python 3.12. The library versions below must match those used to train the model, because the saved model is a pickle.

Windows (PowerShell)

powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass      # only if activation is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt

macOS / Linux

bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-web.txt

requirements-web.txt:

numpy==2.1.3
pandas==2.2.3
scikit-learn==1.6.1
xgboost==3.4.1
shap==0.52.0
joblib==1.6.0
scipy==1.16.3
biopython
flask

First create the model file (see above) and place it at model/venomai_predictor_web.joblib. Then run:

bash
python test_load.py     # should print a prediction
python app.py           # then open the address printed in the terminal

Flask's default address is http://127.0.0.1:5000. The app runs with debug=False on purpose: the debug reloader starts a second process that cannot unpickle the predictor.

Website endpoints
Route	Method	Purpose
/	GET	Landing page
/tool	GET	Tool page
/predict	POST	JSON {"sequence": "..."}, returns target prediction, probabilities, and toxicity reliability
/why	GET	Explanation page (?seq=...)
/explain	GET	SHAP feature contributions for ?seq=...

# Retrain the model in Google Colab

Training needs CD-HIT (Linux) for a valid leakage-aware split, so run it in Colab.

Open a new Colab notebook and paste colab/VenomAI_Colab_v3_2.py into one cell.
Run it. It downloads the data, trains and evaluates all models, runs SHAP, and saves everything under /content/VenomAI_v3/ (datasets, features, results, figures, and models/venomai_predictor.joblib).
To make the lighter copy used by the website, run this in a new cell of the same session:
python
import joblib, os
path = "/content/VenomAI_v3/models/venomai_predictor_web.joblib"
predictor.trained_target_models = {k: v for k, v in predictor.trained_target_models.items() if k != "XGBoost"}
predictor.trained_toxicity_models = {k: v for k, v in predictor.trained_toxicity_models.items() if k != "XGBoost"}
joblib.dump(predictor, path)
print("size:", os.path.getsize(path))
print("XGBoost still inside:", b"xgboost" in open(path, "rb").read().lower())   # must be False
Download the file and compare its size with the number printed in Colab before using it.

If CD-HIT is missing the script falls back to an approximate method and prints a prominent warning. In that case treat every accuracy as an upper bound. Running the training script on native Windows is not recommended.

# Using the predictor from Python

python
import joblib
from venomai_v3 import *          # the predictor class must be importable when unpickling

predictor = joblib.load("model/venomai_predictor_web.joblib")
report = predictor.predict_one("GCKEFGWPCNGEACCS")
print(report["target_prediction"], report["target_confidence"])
print(report["target_probabilities"])

# SHAP explanation of one peptide (Random Forest)

shap_df = predictor.explain_one("GCKEFGWPCNGEACCS", task="target", model_name="Random Forest")
print(shap_df.head(10))

# Data sources and labels

Source	Used for
ArachnoServer	Spider toxin sequences
ConoServer	Cone snail peptides (mature wild type entries only)
UniProt	Reviewed toxin sequences and non-toxin control proteins for spiders, scorpions and cone snails; pathway cross-references

The data is downloaded live when the training script runs, and the databases are updated over time, so a new run can give slightly different counts and scores than the ones reported here. Please check each database's current terms of use and cite them if you reuse the data. Target labels are curated from database text and a documented class-name table, not from independent experiments. Rows whose target cannot be determined are labelled Unknown and are never used for training.

# Limitations

Target labels are text-derived, not experimental ground truth.
Toxicity prediction failed our reliability checks. The non-toxic controls are not experimentally confirmed, and there are only 65 of them.
The model can only answer with its eight target classes. It cannot say "none of these", so peptides with other targets (for example mechanosensitive channels) are forced into the nearest class.
Spider, scorpion and cone snail coverage is uneven. Scorpion data is mostly sodium and potassium channel toxins, and spider data is dominated by enzymes. Among cysteine-rich spider peptides in the training data, about two thirds are labelled sodium channel, so the model can over-predict sodium channel for peptides of that type.
Per-group scores are based on small test sets (42 to 88 peptides) and shift with the data split.
Several target classes are tied to one animal group, which can let the model use taxonomy as a shortcut.
The website uses Random Forest, not the higher-scoring XGBoost (see Troubleshooting).
No temporal or cross-database holdout test yet, and calibration uses ordinary (non-grouped) cross-validation.
UniProt pathway cross-references covered 0% of toxin records in the v3.2 run, so that lookup returned nothing useful.

# Troubleshooting

Problem	Cause and fix
FileNotFoundError: model/venomai_predictor_web.joblib	The model is not in the repository. Train it in Colab and copy the web model into model/ (see above).
EOFError: Ran out of input when loading the model	The .joblib file is empty (often from copy-pasting instead of downloading). Download the real file and compare its size and hash.
XGBoostError: input stream corrupted	The model file still contains an XGBoost model, whose pickle format is fragile across environments. Use the web copy made without XGBoost (see Retrain).
NumPy BitGenerator or similar pickle errors	Library versions differ from training. Install the exact versions in requirements-web.txt.
python points to the wrong version (for example Python 2.7 from another tool)	Use py -3.12 on Windows to create the virtual environment, and install with python -m pip.
PowerShell blocks Activate.ps1	Run Set-ExecutionPolicy -Scope Process Bypass in that window.
Project inside OneDrive behaves strangely	Keep the project outside OneDrive-synced folders.

# Citation

If you use this work, please cite it as:

bibtex
@misc{venomai2026,
  title  = {VenomAI: Explainable Machine Learning for Predicting Venom Peptide Targets and Toxicity},
  author = {Rheashree G and Dhivya Sri S and Priyankha GT},
  year   = {2026},
  note   = {GitHub repository, rheashreegowrishanker/VenomAI}
}

# License

The code in this repository is released under the Apache License 2.0. See the LICENSE file.

The license covers the code only. Data downloaded from ArachnoServer, ConoServer and UniProt remains under those databases' own terms, and the trained model file and any images are not automatically covered either, so check before redistributing them.
