# ============================================================
# VENOMAI v3.2 — SINGLE-FILE COLAB VERSION
# v3.1 = v3.0 + (a) cleaning of contaminated "non-toxic" controls,
# (b) a length+cysteine-only baseline, (c) a matched "hard" toxicity
# test, (d) a stricter toxicity reliability flag. See sections marked v3.1.
# v3.2 = v3.1 + ConoServer fixes (see sections marked v3.2):
# (a) the pharmacological-family / cysteine-framework header fields were
#     assigned in swapped order, now corrected;
# (b) ConoServer "Precursor", "Patent" and "Synthetic" entries are no
#     longer used (precursors carry signal/pro-regions, the others are
#     not natural mature peptides) -- only "Wild type" entries are kept;
# (c) a documented, curated rule table maps ConoServer toxin classes
#     (e.g. conantokin) and pharmacological families (e.g. alpha
#     conotoxin) to target classes, used ONLY when the text rules found
#     nothing.
# Explainable ML for Venom Peptide Target and Toxicity Prediction
# ============================================================
#
# HOW TO USE IN GOOGLE COLAB
# ---------------------------
# 1. Open a new Colab notebook (colab.research.google.com).
# 2. Paste this ENTIRE file into one cell.
# 3. Run the cell (Shift+Enter). It installs what it needs, then runs
#    the full pipeline: downloads data, trains models, explains them,
#    and saves everything under /content/VenomAI_v3/.
# 4. To just score a new peptide later without retraining, see the
#    very bottom of this file ("ONE-OFF PREDICTION EXAMPLE").
#
# WHAT THIS VERSION FIXES vs. the original single script
# ---------------------------------------------------------
# 1. SHAP now explains ALL FOUR trained models (Random Forest, SVM,
#    XGBoost, Neural Network) -- not just Random Forest. Tree models
#    get exact TreeExplainer SHAP; SVM/Neural Network use a sampled
#    generic explainer (slower per-sample, so it runs on a capped
#    background + capped explained set -- documented, not hidden).
# 2. Richer features: the original AAC + dipeptide + physicochemical
#    set is joined by 84 CTD (Composition/Transition/Distribution)
#    descriptors across 4 physicochemical properties. An optional,
#    off-by-default hook for ESM-2 protein-language-model embeddings
#    is included for anyone running with GPU/internet to spare.
# 3. Toxicity negative class is now REAL: genuine non-toxin-keyword,
#    reviewed, length-matched UniProt controls from the same taxa,
#    not just text mined out of an already-toxin-only pool.
# 4. Neural Network added (the abstract promised it; the original
#    script never had one).
# 5. Hyperparameters are tuned (RandomizedSearchCV, leakage-aware
#    inner CV) instead of fixed.
# 6. Probabilities are calibrated before being reported as
#    "confidence" or used in the priority score.
# 7. Real pathway cross-references (KEGG/Reactome) are pulled from
#    UniProt directly -- clearly labeled as a lookup, never as an ML
#    prediction.
# 8. A target/toxicity "reliability" flag (driven by each model's own
#    CV balanced accuracy) is threaded through to every prediction,
#    so an unreliable model can't quietly be trusted downstream.
#
# KNOWN LIMITATIONS KEPT AS-IS (see printed report at the end):
# - Target labels are still text-derived from curator annotation, not
#   independent experimental ground truth.
# - Taxonomic coverage is limited to spiders, scorpions, cone snails.
# - No temporal or cross-database holdout test yet.
# - Calibration uses ordinary (non-grouped) CV internally.
# - Pathway coverage depends on how many sequences already have a
#   KEGG/Reactome cross-reference in UniProt.
# ============================================================

# ============================================================
# SECTION 0 — INSTALL / IMPORT
# ============================================================
import os, re, io, gzip, json, hashlib, urllib.parse, warnings
import subprocess, sys, shutil, time
from datetime import datetime, timezone


def _install_if_missing():
    packages = {
        "Bio": "biopython==1.83",
        "xgboost": "xgboost==2.0.3",
        "shap": "shap==0.45.0",
        "bs4": "beautifulsoup4==4.12.3",
        "sklearn": "scikit-learn==1.4.2",
        "pandas": "pandas==2.2.2",
        "numpy": "numpy==1.26.4",
        "requests": "requests==2.31.0",
        "matplotlib": "matplotlib==3.8.4",
        "joblib": "joblib==1.4.0",
    }
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


_install_if_missing()


def _install_cdhit():
    """CD-HIT gives exact sequence-identity clustering for leakage
    control. If it can't be installed, we fall back to an approximate
    method and say so LOUDLY (not a quiet log line) because it
    weakens how much the reported accuracy numbers can be trusted."""
    if shutil.which("cd-hit"):
        return True
    try:
        subprocess.run(["apt-get", "install", "-y", "-q", "cd-hit"],
                        check=True, capture_output=True, text=True, timeout=180)
        return shutil.which("cd-hit") is not None
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cd-hit"],
                            check=True, capture_output=True, text=True, timeout=180)
        except Exception:
            pass
        return shutil.which("cd-hit") is not None


CDHIT_AVAILABLE = _install_cdhit()
if not CDHIT_AVAILABLE:
    print("=" * 78)
    print("WARNING: CD-HIT could not be installed. Falling back to an")
    print("APPROXIMATE clustering method for leakage control. Treat any")
    print("reported accuracy as an UPPER BOUND until re-run with real CD-HIT.")
    print("=" * 78)

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
import shap

from bs4 import BeautifulSoup
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.model_selection import (
    StratifiedGroupKFold, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1 — CONFIG
# All tunable constants in one place. Change behaviour here.
# ============================================================
class config:
    RANDOM_STATE = 42

    BASE_DIR = "/content/VenomAI_v3"
    RAW_DIR = os.path.join(BASE_DIR, "raw_data")
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    FEATURE_DIR = os.path.join(BASE_DIR, "features")
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    FIGURE_DIR = os.path.join(BASE_DIR, "figures")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    ALL_DIRS = [BASE_DIR, RAW_DIR, PROCESSED_DIR, FEATURE_DIR,
                RESULT_DIR, MODEL_DIR, FIGURE_DIR, LOG_DIR]

    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    MIN_SEQUENCE_LENGTH = 5
    MAX_SEQUENCE_LENGTH = 5000

    MIN_TARGET_CLASS_SIZE = 10
    MIN_TOXICITY_CLASS_SIZE = 20
    MIN_RELIABLE_BALANCED_ACCURACY = 0.60

    CDHIT_IDENTITY = 0.70

    MAX_SELECTED_FEATURES = 150

    # Off by default: needs a model download + is much faster with a
    # GPU, neither of which is assumed here. Flip to True if you have
    # both in your Colab runtime.
    USE_ESM_EMBEDDINGS = False
    ESM_MODEL_NAME = "esm2_t6_8M_UR50D"

    RANDOM_SEARCH_ITER = 15
    RANDOM_SEARCH_INNER_SPLITS = 3

    SHAP_BACKGROUND_SIZE = 50
    SHAP_EXPLAIN_SAMPLE_SIZE = 40

    NEGATIVE_CONTROL_LENGTH_MIN = 5
    NEGATIVE_CONTROL_LENGTH_MAX = 200

    # ---- v3.1: cleaning the "non-toxic" controls ----
    # Some controls are venom components that simply lack UniProt's
    # exact "toxin" keyword (e.g. "neurotoxin bmkx-a1-s31",
    # "venom peptide ocy4", "phospholipase A2"). Any control whose name,
    # function text or keywords contain one of these terms is removed
    # from the toxicity labels. This is a NAME-based filter, so it will
    # miss some -- the homology filter below catches more.
    CONTROL_EXCLUDE_TERMS = [
        "venom", "toxin", "latrotoxin", "latrodectin", "antimicrobial",
        "phospholipase", "conopeptide", "conopressin", "contryphan",
        "ndbp", "non-disulfide", "cryptide", "insecticid", "defensin",
        "kunitz",
    ]

    # ---- v3.1: honest toxicity evaluation ----
    # The models must beat a trivial "length + cysteine fraction only"
    # model by at least this much balanced accuracy, otherwise the
    # toxicity model is flagged unreliable.
    MIN_GAIN_OVER_SIMPLE_BASELINE = 0.05
    # "Matched" test: toxic and non-toxic examples are sampled to have
    # the same number of examples in every (length bin x cysteine bin)
    # cell, so length and cysteine content can no longer give the answer.
    MATCH_LENGTH_BIN_EDGES = [0, 20, 40, 70, 100000]
    MATCH_CYS_BIN_EDGES = [-1, 0.0, 0.08, 0.14, 1.0]

    # ---- v3.2: ConoServer data quality ----
    # ConoServer "protein_type" values seen in the real download:
    # Precursor (signal + pro-region + peptide, ~73 aa), Wild type
    # (natural mature peptide, ~32 aa), Patent and Synthetic (engineered /
    # non-natural). The website scores MATURE peptides, so only these
    # types are kept. Add "Synthetic" here if you deliberately want analogs.
    CONOSERVER_KEEP_PROTEIN_TYPES = ["Wild type"]


np.random.seed(config.RANDOM_STATE)
for d in config.ALL_DIRS:
    os.makedirs(d, exist_ok=True)


# ============================================================
# SECTION 2 — HTTP HELPER
# ============================================================
def safe_get(url, params=None, timeout=90, retries=3):
    headers = {"User-Agent": "VenomAI/3.0 academic bioinformatics project"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise last_error


# ============================================================
# SECTION 3 — SEQUENCE UTILITIES
# ============================================================
def clean_sequence(sequence):
    if sequence is None:
        return None
    try:
        if pd.isna(sequence):
            return None
    except (TypeError, ValueError):
        pass
    sequence = re.sub(r"\s+", "", str(sequence).upper())
    sequence = sequence.replace("*", "")
    if not sequence:
        return None
    if set(sequence) - config.VALID_AA:
        return None
    if not (config.MIN_SEQUENCE_LENGTH <= len(sequence) <= config.MAX_SEQUENCE_LENGTH):
        return None
    return sequence


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def parse_feature_ranges(feature_text):
    if feature_text is None:
        return []
    try:
        if pd.isna(feature_text):
            return []
    except (TypeError, ValueError):
        pass
    text = str(feature_text)
    ranges = []
    for match in re.finditer(r"(\d+)\.\.(\d+)", text):
        start, end = int(match.group(1)), int(match.group(2))
        ranges.append((start, end, text))
    return ranges


def extract_mature_sequence(full_sequence, peptide_features, chain_features):
    """Prefer annotated PEPTIDE/CHAIN regions over the full precursor
    sequence; fall back to the cleaned full sequence if nothing is
    annotated."""
    full_sequence = clean_sequence(full_sequence)
    if full_sequence is None:
        return None, "invalid"

    candidates = []
    for priority, feature_text in [(0, peptide_features), (1, chain_features)]:
        for start, end, raw in parse_feature_ranges(feature_text):
            if start < 1 or end > len(full_sequence) or start > end:
                continue
            note = raw.lower()
            score = sum(term in note for term in ["mature", "toxin", "venom", "peptide", "chain"])
            score += (2 - priority)
            seq = full_sequence[start - 1:end]
            if len(seq) >= config.MIN_SEQUENCE_LENGTH:
                candidates.append((score, len(seq), start, end, seq))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][4], "annotated_mature_region"

    return full_sequence, "full_sequence_fallback"


# ============================================================
# SECTION 4 — DATA SOURCES (ArachnoServer / ConoServer / UniProt)
# ============================================================
ARACHNO_DOWNLOAD_PAGE = "https://www.arachnoserver.org/downloadsequences.html"
ARACHNO_FALLBACK_FASTA = "https://www.arachnoserver.org/download/ArachnoServer_all.fasta"
CONOSERVER_DOWNLOAD_PAGE = "https://www.conoserver.org/?page=download"
UNIPROT_ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"

# xref_kegg / xref_reactome give REAL pathway cross-references from
# UniProt curation -- see build_pathway_crossreferences() below.
UNIPROT_FIELDS = [
    "accession", "id", "protein_name", "organism_name",
    "organism_id", "length", "sequence", "reviewed", "fragment",
    "cc_function", "cc_toxic_dose", "keyword", "go_f",
    "ft_peptide", "ft_chain", "lit_pubmed_id", "date_modified",
    "xref_kegg", "xref_reactome",
]

EXPECTED_COLUMNS = [
    "record_id", "sequence", "sequence_length", "source_database",
    "venom_group", "source_category", "species", "protein_name",
    "function_annotation", "toxicity_annotation", "keywords",
    "go_molecular_function", "peptide_features", "chain_features",
    "conopeptide_name", "protein_type", "toxin_class", "gene_superfamily",
    "cysteine_framework", "pharmacological_family", "evidence",
    "uniprot_id", "reviewed", "fragment", "sequence_type",
    "precursor_sequence", "header", "xref_kegg", "xref_reactome",
]

POSITIVE_UNIPROT_QUERIES = [
    ("reviewed:true AND keyword:toxin AND taxonomy_name:Araneae", "Spider"),
    ("reviewed:true AND keyword:toxin AND taxonomy_name:Scorpiones", "Scorpion"),
    ("reviewed:true AND keyword:toxin AND taxonomy_name:Conus", "Cone_snail"),
]

# The fix for "the toxicity model learns nothing": real non-toxin,
# length-matched, reviewed controls from the SAME taxa.
NEGATIVE_UNIPROT_QUERIES = [
    (f"reviewed:true NOT keyword:toxin AND taxonomy_name:Araneae "
     f"AND length:[{config.NEGATIVE_CONTROL_LENGTH_MIN} TO {config.NEGATIVE_CONTROL_LENGTH_MAX}]", "Spider"),
    (f"reviewed:true NOT keyword:toxin AND taxonomy_name:Scorpiones "
     f"AND length:[{config.NEGATIVE_CONTROL_LENGTH_MIN} TO {config.NEGATIVE_CONTROL_LENGTH_MAX}]", "Scorpion"),
    (f"reviewed:true NOT keyword:toxin AND taxonomy_name:Conus "
     f"AND length:[{config.NEGATIVE_CONTROL_LENGTH_MIN} TO {config.NEGATIVE_CONTROL_LENGTH_MAX}]", "Cone_snail"),
]


def fasta_to_dataframe(fasta_text, source_database, venom_group):
    rows = []
    for record in SeqIO.parse(io.StringIO(fasta_text), "fasta"):
        seq = clean_sequence(str(record.seq))
        if seq is None:
            continue
        rows.append({
            "record_id": record.id, "header": record.description, "sequence": seq,
            "sequence_length": len(seq), "source_database": source_database,
            "venom_group": venom_group, "source_category": "toxin_positive",
        })
    return pd.DataFrame(rows)


def fetch_arachnoserver():
    fasta_url = None
    try:
        page = safe_get(ARACHNO_DOWNLOAD_PAGE)
        soup = BeautifulSoup(page.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(" ", strip=True).lower()
            if ("all peptide sequences" in text or "all.pep.fa" in href.lower()
                    or href.lower().endswith((".fa", ".fasta", ".faa"))):
                fasta_url = urllib.parse.urljoin(ARACHNO_DOWNLOAD_PAGE, href)
                if "nucleic" not in href.lower():
                    break
    except Exception as exc:
        print("ArachnoServer page scrape failed, trying fallback URL:", exc)

    if fasta_url is None:
        fasta_url = ARACHNO_FALLBACK_FASTA

    response = safe_get(fasta_url)
    raw_path = os.path.join(config.RAW_DIR, "arachnoserver_all_peptides.fasta")
    with open(raw_path, "w") as fh:
        fh.write(response.text)
    return fasta_to_dataframe(response.text, "ArachnoServer", "Spider")


def parse_conoserver_header(header):
    # v3.2: the real file order is ... gene_superfamily | pharmacological
    # family | cysteine framework | evidence. Verified on real headers:
    # A superfamily -> framework I, O1 -> VI/VII, M -> III, T -> V.
    # (v3.0/v3.1 had the two middle names swapped.)
    fields = ["conoserver_id", "conopeptide_name", "species", "protein_type",
              "toxin_class", "gene_superfamily", "pharmacological_family",
              "cysteine_framework", "evidence"]
    parts = str(header).split("|")
    return {f: (parts[i].strip() if i < len(parts) else np.nan) for i, f in enumerate(fields)}


def fetch_conoserver():
    page = safe_get(CONOSERVER_DOWNLOAD_PAGE)
    soup = BeautifulSoup(page.text, "html.parser")

    fasta_url = None
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "conoserver_protein.fa.gz" in href.lower():
            fasta_url = urllib.parse.urljoin(CONOSERVER_DOWNLOAD_PAGE, href)
            break
    if fasta_url is None:
        raise RuntimeError("Could not locate ConoServer protein FASTA.")

    response = safe_get(fasta_url)
    try:
        text = gzip.decompress(response.content).decode("utf-8", errors="ignore")
    except OSError:
        text = response.text

    raw_path = os.path.join(config.RAW_DIR, "conoserver_protein.fasta")
    with open(raw_path, "w") as fh:
        fh.write(text)

    df = fasta_to_dataframe(text, "ConoServer", "Cone_snail")
    if not df.empty:
        annotations = pd.DataFrame([parse_conoserver_header(x) for x in df["header"]])
        df = pd.concat([df.reset_index(drop=True), annotations.reset_index(drop=True)], axis=1)
    return df


def query_uniprot(query, group_label, source_category, page_size=500):
    params = {"query": query, "format": "tsv", "fields": ",".join(UNIPROT_FIELDS), "size": page_size}
    response = safe_get(UNIPROT_ENDPOINT, params=params)
    total = int(response.headers.get("x-total-results", "0"))
    print(f"UniProt [{source_category}] {group_label}: {total} records reported.")

    frames = []
    while True:
        df = pd.read_csv(io.StringIO(response.text), sep="\t")
        if df.empty:
            break
        frames.append(df)

        link_header = response.headers.get("Link")
        next_url = None
        if link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    m = re.search(r"<([^>]+)>", part)
                    if m:
                        next_url = m.group(1)
                    break
        if not next_url:
            break
        response = safe_get(next_url)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    rename_map = {
        "Entry": "uniprot_id", "Entry Name": "entry_name", "Protein names": "protein_name",
        "Organism": "species", "Organism ID": "organism_id", "Length": "sequence_length",
        "Sequence": "sequence", "Reviewed": "reviewed", "Fragment": "fragment",
        "Function [CC]": "function_annotation", "Toxic dose": "toxicity_annotation",
        "Keywords": "keywords", "Gene Ontology (molecular function)": "go_molecular_function",
        "Peptide": "peptide_features", "Chain": "chain_features", "PubMed ID": "pubmed_ids",
        "Date of last modification": "date_modified", "KEGG": "xref_kegg", "Reactome": "xref_reactome",
    }
    df = df.rename(columns=rename_map)
    if "sequence" not in df.columns:
        raise RuntimeError("UniProt response contained no sequence field.")

    mature_sequences, sequence_types = [], []
    for _, row in df.iterrows():
        seq, seq_type = extract_mature_sequence(row.get("sequence"), row.get("peptide_features"), row.get("chain_features"))
        mature_sequences.append(seq)
        sequence_types.append(seq_type)

    df["precursor_sequence"] = df["sequence"].apply(clean_sequence)
    df["sequence"] = mature_sequences
    df["sequence_type"] = sequence_types
    df = df[df["sequence"].notna()].copy()
    df["venom_group"] = group_label
    df["source_database"] = "UniProt"
    df["source_category"] = source_category
    df["record_id"] = df["uniprot_id"]

    safe_label = re.sub(r"[^a-z0-9]+", "_", group_label.lower())
    raw_path = os.path.join(config.RAW_DIR, f"uniprot_{source_category}_{safe_label}.tsv")
    df.to_csv(raw_path, sep="\t", index=False)
    return df


CONOSERVER_FILTER_SUMMARY = {}


def filter_conoserver_natural_mature(df):
    """v3.2: keep only natural MATURE ConoServer entries (protein_type in
    config.CONOSERVER_KEEP_PROTEIN_TYPES). Precursors contain signal and
    pro-regions (so their sequences do not look like the mature peptides
    users paste into the website, and conserved signal peptides can leak
    the gene superfamily); Patent / Synthetic entries are not natural."""
    global CONOSERVER_FILTER_SUMMARY
    if df is None or df.empty or "protein_type" not in df.columns:
        return df
    before = df["protein_type"].value_counts(dropna=False).to_dict()
    keep = df["protein_type"].isin(config.CONOSERVER_KEEP_PROTEIN_TYPES)
    kept = df[keep].copy()
    CONOSERVER_FILTER_SUMMARY = {
        "kept_protein_types": list(config.CONOSERVER_KEEP_PROTEIN_TYPES),
        "rows_before": int(len(df)), "rows_kept": int(len(kept)),
        "protein_types_before": {str(k): int(v) for k, v in before.items()},
    }
    print("\nCONOSERVER FILTER (v3.2)")
    print(f"  rows before: {len(df)}  kept: {len(kept)}  "
          f"(protein types kept: {config.CONOSERVER_KEEP_PROTEIN_TYPES})")
    print(f"  protein types before filtering: {CONOSERVER_FILTER_SUMMARY['protein_types_before']}")
    return kept


def collect_data():
    datasets = []

    try:
        datasets.append(fetch_arachnoserver())
        print("ArachnoServer downloaded.")
    except Exception as exc:
        print("ArachnoServer warning:", exc)

    for query, group in POSITIVE_UNIPROT_QUERIES:
        try:
            datasets.append(query_uniprot(query, group, "toxin_positive"))
            print(f"UniProt {group} (positive) downloaded.")
        except Exception as exc:
            print(f"UniProt {group} positive warning:", exc)

    try:
        datasets.append(filter_conoserver_natural_mature(fetch_conoserver()))
        print("ConoServer downloaded.")
    except Exception as exc:
        print("ConoServer warning:", exc)

    for query, group in NEGATIVE_UNIPROT_QUERIES:
        try:
            datasets.append(query_uniprot(query, group, "non_toxin_control"))
            print(f"UniProt {group} (negative control) downloaded.")
        except Exception as exc:
            print(f"UniProt {group} negative-control warning:", exc)

    datasets = [d for d in datasets if d is not None and not d.empty]
    if not datasets:
        raise RuntimeError("No dataset was retrieved. Check internet access.")
    return datasets


def standardize(df):
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "source_category" in df.columns:
        df["source_category"] = df["source_category"].fillna("toxin_positive")
    return df[EXPECTED_COLUMNS].copy()


# ============================================================
# SECTION 5 — LABELING (target class + toxicity, text-rule based)
# Still a TEXT-DERIVED CURATION LAYER, not independent experimental
# ground truth -- that's a kept limitation, not hidden. "Unknown" is
# never silently converted into a training label.
# ============================================================
TARGET_RULES = {
    "Voltage-gated sodium channel": [r"voltage[- ]gated sodium", r"\bsodium channel\b", r"\bnav\b", r"na\(v\)"],
    "Potassium channel": [r"voltage[- ]gated potassium", r"\bpotassium channel\b", r"\bkv\b", r"k\(v\)"],
    "Calcium channel": [r"voltage[- ]gated calcium", r"\bcalcium channel\b", r"\bcav\b", r"ca\(v\)"],
    "Nicotinic acetylcholine receptor": [r"nicotinic acetylcholine receptor", r"nicotinic receptor"],
    "Glutamate receptor": [r"glutamate receptor", r"ionotropic glutamate receptor"],
    "G-protein-coupled receptor": [r"g[- ]protein[- ]coupled receptor", r"\bgpcr\b"],
    "Enzyme": [r"phospholipase", r"protease", r"metalloprotease", r"serine protease", r"enzyme inhibitor", r"enzyme target"],
    "Membrane-associated target": [r"membrane protein", r"cell membrane", r"membrane"],
}

# ---- v3.2: curated ConoServer rule tables -------------------------------
# Used ONLY when the text rules above found nothing (never to override
# them). Each line is expert knowledge about conotoxin pharmacology, written
# down so it can be checked and cited -- it is NOT per-peptide hand labelling.
# VERIFY each line against ConoServer's documentation / the primary
# literature before citing it in the report.
CONOSERVER_CLASS_RULES = {          # key = ConoServer "toxin_class"
    "conantokin": "Glutamate receptor",            # NMDA receptor antagonists
    "con-ikot-ikot": "Glutamate receptor",         # AMPA receptor modulators
    "conodipine": "Enzyme",                        # phospholipase A2
    "conohyal": "Enzyme",                          # hyaluronidase
    "contulakin": "G-protein-coupled receptor",    # neurotensin receptor agonist
    "conopressin": "G-protein-coupled receptor",   # vasopressin/oxytocin-type receptors
    "conoporin": "Membrane-associated target",     # pore-forming
    # "conkunitzin" -> "Potassium channel" is deliberately NOT included:
    # extending Kv block from the best-studied member to the whole class
    # is an inference. Uncomment only if you are comfortable defending it.
}
CONOSERVER_PHARMACOLOGY_RULES = {   # key = ConoServer "pharmacological_family"
    "alpha conotoxin": "Nicotinic acetylcholine receptor",
    "omega conotoxin": "Calcium channel",
    "delta conotoxin": "Voltage-gated sodium channel",
    "mu conotoxin": "Voltage-gated sodium channel",
    # "chi conotoxin" (norepinephrine transporter) has no matching class here.
}


def conoserver_curated_label(row):
    """Returns a target class from the curated ConoServer tables, or None."""
    def clean(value):
        return "" if pd.isna(value) else str(value).strip().lower()

    pharm = clean(row.get("pharmacological_family"))
    if pharm in CONOSERVER_PHARMACOLOGY_RULES:
        return CONOSERVER_PHARMACOLOGY_RULES[pharm]
    tclass = clean(row.get("toxin_class"))
    if tclass in CONOSERVER_CLASS_RULES:
        return CONOSERVER_CLASS_RULES[tclass]
    return None

NEGATIVE_TOXICITY_TERMS = ["non-toxic", "nontoxic", "non toxic", "no toxic activity", "not toxic"]
POSITIVE_TOXICITY_TERMS = ["toxic dose", "toxicity", "toxic activity", "lethal dose", "ld50", "lc50", "lethal"]


def infer_target(row):
    if row.get("source_category") != "toxin_positive":
        return "Not_applicable", "Negative control record", 0

    fields = ["function_annotation", "protein_name", "keywords",
              "go_molecular_function", "pharmacological_family", "toxin_class"]
    text = " ".join(str(row.get(c, "")) for c in fields if not pd.isna(row.get(c, np.nan))).lower()

    matches = []
    for target, patterns in TARGET_RULES.items():
        hits = sum(bool(re.search(p, text)) for p in patterns)
        if hits:
            matches.append((target, hits))

    if not matches:
        curated = conoserver_curated_label(row)          # v3.2
        if curated is not None:
            return curated, "ConoServer curated class rule", 1
        return "Unknown", "No target rule matched", 0

    matches.sort(key=lambda x: x[1], reverse=True)
    if len(matches) > 1 and matches[0][1] == matches[1][1]:
        curated = conoserver_curated_label(row)          # v3.2
        if curated is not None and curated in [m[0] for m in matches]:
            return curated, "ConoServer curated class rule (tie-break)", 1
        return "Unknown", "Ambiguous target annotation", 0

    return matches[0][0], "Text-derived curated rule", matches[0][1]


def infer_toxicity(row):
    """Combines two independent sources of evidence: (a) explicit
    toxic/non-toxic TEXT evidence within the toxin-positive pool, and
    (b) records pulled specifically as non-toxin-keyword, reviewed,
    length-matched controls -- labeled Non_toxic by construction of
    the query, not by keyword search inside them."""
    if row.get("source_category") == "non_toxin_control":
        return "Non_toxic", "Non-toxin-keyword control record (query-level negative)"

    text = " ".join(
        str(row.get(c, "")) for c in ["toxicity_annotation", "function_annotation"]
        if not pd.isna(row.get(c, np.nan))
    ).lower()

    if any(term in text for term in NEGATIVE_TOXICITY_TERMS):
        return "Non_toxic", "Explicit negative text (toxin-positive source)"
    if any(term in text for term in POSITIVE_TOXICITY_TERMS):
        return "Toxic", "Explicit positive toxicity text"

    return "Unknown", "No explicit toxicity evidence"


def label_master(master):
    target_result = master.apply(infer_target, axis=1, result_type="expand")
    master["target_class"] = target_result[0]
    master["target_label_source"] = target_result[1]
    master["target_rule_strength"] = target_result[2]

    tox_result = master.apply(infer_toxicity, axis=1, result_type="expand")
    master["toxicity_status"] = tox_result[0]
    master["toxicity_label_source"] = tox_result[1]
    return master


def flag_contaminated_controls(master):
    """v3.1: remove "non-toxic" controls that are probably not safe.

    A control is dropped from the toxicity labels (set to Unknown, never
    to Toxic) if EITHER
      (a) its name / function text / keywords contain a venom- or
          toxin-like term (config.CONTROL_EXCLUDE_TERMS), OR
      (b) it sits in the same 70%-identity cluster as a known toxin,
          i.e. it is a close relative of a toxin.
    Must run AFTER clustering (needs master["sequence_cluster"]).
    Returns (master, counts_dict).
    """
    is_control = master["source_category"] == "non_toxin_control"

    text = (master["protein_name"].fillna("").astype(str) + " "
            + master["function_annotation"].fillna("").astype(str) + " "
            + master["keywords"].fillna("").astype(str)).str.lower()
    pattern = "|".join(re.escape(t) for t in config.CONTROL_EXCLUDE_TERMS)
    name_hit = is_control & text.str.contains(pattern, regex=True, na=False)

    toxin_clusters = set(master.loc[master["source_category"] == "toxin_positive", "sequence_cluster"])
    homology_hit = is_control & master["sequence_cluster"].isin(toxin_clusters)

    flagged = name_hit | homology_hit
    master.loc[name_hit, "toxicity_label_source"] = "Excluded control: venom/toxin-like name or annotation"
    master.loc[homology_hit & ~name_hit, "toxicity_label_source"] = "Excluded control: close relative (>=70% identity) of a known toxin"
    master.loc[flagged, "toxicity_status"] = "Unknown"

    counts = {
        "controls_before": int(is_control.sum()),
        "removed_by_name": int(name_hit.sum()),
        "removed_by_homology_only": int((homology_hit & ~name_hit).sum()),
        "controls_kept": int((is_control & ~flagged).sum()),
    }
    print("\nNEGATIVE-CONTROL CLEANING (v3.1)")
    print(f"  controls before cleaning        : {counts['controls_before']}")
    print(f"  removed (venom/toxin-like name) : {counts['removed_by_name']}")
    print(f"  removed (relative of a toxin)   : {counts['removed_by_homology_only']}")
    print(f"  controls kept as Non_toxic      : {counts['controls_kept']}")
    return master, counts


# ============================================================
# SECTION 6 — SEQUENCE-SIMILARITY CLUSTERING (leakage control)
# ============================================================
def write_fasta(df, path):
    with open(path, "w") as fh:
        for i, seq in enumerate(df["sequence"]):
            fh.write(f">SEQ_{i}\n{seq}\n")


def parse_cdhit_clusters(clstr_path, n):
    groups = {}
    current_cluster = None
    with open(clstr_path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = line.replace(">Cluster", "").strip()
                continue
            m = re.search(r">SEQ_(\d+)\.\.\.", line)
            if m and current_cluster is not None:
                groups[int(m.group(1))] = f"cluster_{current_cluster}"
    return [groups.get(i, f"singleton_{i}") for i in range(n)]


def sequence_similarity_groups(df):
    """Returns (groups, method_label, is_strict_cdhit)."""
    fasta_path = os.path.join(config.PROCESSED_DIR, "clustering_input.fasta")
    write_fasta(df.reset_index(drop=True), fasta_path)

    cdhit = shutil.which("cd-hit")
    if cdhit:
        output = os.path.join(config.PROCESSED_DIR, "cdhit_70")
        cmd = [cdhit, "-i", fasta_path, "-o", output, "-c", str(config.CDHIT_IDENTITY),
               "-n", "4", "-d", "0", "-T", "2", "-M", "0", "-g", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        clstr = output + ".clstr"
        if result.returncode == 0 and os.path.exists(clstr):
            return parse_cdhit_clusters(clstr, len(df)), "CD-HIT 70% identity", True

    def fingerprint(seq):
        return tuple(sorted(set(seq[i:i + 3] for i in range(len(seq) - 2))))

    fingerprints, groups = {}, []
    for seq in df["sequence"]:
        fp = fingerprint(seq)
        key = hashlib.sha1("|".join(fp).encode()).hexdigest()[:16]
        if key not in fingerprints:
            fingerprints[key] = f"approx_cluster_{len(fingerprints)}"
        groups.append(fingerprints[key])

    return groups, "Approximate 3-mer fingerprint fallback", False


# ============================================================
# SECTION 7 — FEATURE ENGINEERING
# Classic AAC/dipeptide/physicochemical + NEW: CTD descriptors
# (Composition/Transition/Distribution). Optional ESM-2 hook (off by
# default).
# ============================================================
AMINO_ACIDS = config.AMINO_ACIDS


def amino_acid_composition(seq):
    n = len(seq)
    return {f"AAC_{aa}": seq.count(aa) / n for aa in AMINO_ACIDS}


def dipeptide_composition(seq):
    result = {}
    total = len(seq) - 1
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dp = aa1 + aa2
            count = sum(1 for i in range(total) if seq[i:i + 2] == dp) if total > 0 else 0
            result[f"DP_{dp}"] = count / total if total > 0 else 0.0
    return result


def sequence_complexity(seq):
    counts = np.array([seq.count(aa) for aa in AMINO_ACIDS])
    probs = counts[counts > 0] / len(seq)
    return float(-(probs * np.log2(probs)).sum())


def cysteine_spacing(seq):
    positions = [i for i, aa in enumerate(seq, start=1) if aa == "C"]
    if len(positions) < 2:
        return {"cys_count": len(positions), "cys_mean_spacing": 0.0,
                "cys_max_spacing": 0.0, "cys_min_spacing": 0.0}
    gaps = np.diff(positions)
    return {"cys_count": len(positions), "cys_mean_spacing": float(np.mean(gaps)),
            "cys_max_spacing": float(np.max(gaps)), "cys_min_spacing": float(np.min(gaps))}


def physicochemical_features(seq):
    p = ProteinAnalysis(seq)
    n = len(seq)
    cys = cysteine_spacing(seq)
    return {
        "length": n,
        "molecular_weight": p.molecular_weight(),
        "isoelectric_point": p.isoelectric_point(),
        "gravy": p.gravy(),
        "aromaticity": p.aromaticity(),
        "instability_index": p.instability_index(),
        "aliphatic_index": (2.9 * seq.count("A") / n + 3.9 * seq.count("V") / n +
                             3.9 * seq.count("I") / n + 3.9 * seq.count("L") / n),
        "cysteine_fraction": seq.count("C") / n,
        "basic_fraction": sum(seq.count(a) for a in "KRH") / n,
        "acidic_fraction": sum(seq.count(a) for a in "DE") / n,
        "charged_fraction": sum(seq.count(a) for a in "DEKR") / n,
        "hydrophobic_fraction": sum(seq.count(a) for a in "AVILMFWY") / n,
        "sequence_entropy": sequence_complexity(seq),
        **cys,
    }


# CTD property groupings (Dubchak-style). Each of the 20 amino acids
# falls into exactly one of 3 groups per property.
PROPERTY_GROUPS = {
    "hydrophobicity": {1: set("RKEDQN"), 2: set("GASTPHY"), 3: set("CLVIMFW")},
    "charge": {1: set("KR"), 2: set("DE"), 3: set("ACFGHILMNPQSTVWY")},
    "polarity": {1: set("LIFWCMVY"), 2: set("PATGS"), 3: set("HQRKNED")},
    "secondary_structure": {1: set("EALMQKRH"), 2: set("VIYCWFT"), 3: set("GNPSD")},
}


def _group_lookup(groups):
    lookup = {}
    for g, residues in groups.items():
        for aa in residues:
            lookup[aa] = g
    return lookup


def ctd_composition(seq, groups):
    n = len(seq)
    return {f"C{g}": sum(seq.count(aa) for aa in residues) / n for g, residues in groups.items()}


def ctd_transition(seq, groups):
    group_of = _group_lookup(groups)
    pairs = [(1, 2), (1, 3), (2, 3)]
    counts = {p: 0 for p in pairs}
    total = len(seq) - 1
    for i in range(total):
        g1, g2 = group_of[seq[i]], group_of[seq[i + 1]]
        if g1 == g2:
            continue
        pair = tuple(sorted((g1, g2)))
        counts[pair] += 1
    return {f"T{p[0]}{p[1]}": (counts[p] / total if total > 0 else 0.0) for p in pairs}


def ctd_distribution(seq, groups):
    group_of = _group_lookup(groups)
    n = len(seq)
    result = {}
    for g in groups:
        positions = [i + 1 for i, aa in enumerate(seq) if group_of[aa] == g]
        count = len(positions)
        for pct, label in [(0.0, "0"), (0.25, "25"), (0.5, "50"), (0.75, "75"), (1.0, "100")]:
            if count == 0:
                result[f"D{g}_{label}"] = 0.0
                continue
            idx = max(0, int(np.ceil(pct * count)) - 1) if pct > 0 else 0
            idx = min(idx, count - 1)
            result[f"D{g}_{label}"] = positions[idx] / n
    return result


def ctd_features(seq):
    """~84 features: 4 properties x (3 composition + 3 transition + 15 distribution)."""
    feat = {}
    for prop_name, groups in PROPERTY_GROUPS.items():
        for k, v in ctd_composition(seq, groups).items():
            feat[f"CTD_{prop_name}_{k}"] = v
        for k, v in ctd_transition(seq, groups).items():
            feat[f"CTD_{prop_name}_{k}"] = v
        for k, v in ctd_distribution(seq, groups).items():
            feat[f"CTD_{prop_name}_{k}"] = v
    return feat


_esm_model_cache = {}


def esm_embedding_features(seq):
    """Optional ESM-2 embedding hook, off by default (config.USE_ESM_EMBEDDINGS).
    Best-effort: returns {} if the packages/model aren't available
    rather than crashing the run."""
    if not config.USE_ESM_EMBEDDINGS:
        return {}
    try:
        import torch
        import esm  # fair-esm package
    except ImportError:
        return {}
    try:
        if "model" not in _esm_model_cache:
            model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM_MODEL_NAME)
            model.eval()
            _esm_model_cache["model"] = model
            _esm_model_cache["batch_converter"] = alphabet.get_batch_converter()
        model = _esm_model_cache["model"]
        batch_converter = _esm_model_cache["batch_converter"]
        _, _, tokens = batch_converter([("query", seq)])
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers])
        representations = out["representations"][model.num_layers][0, 1:len(seq) + 1]
        mean_repr = representations.mean(0).numpy()
        return {f"ESM_{i}": float(v) for i, v in enumerate(mean_repr)}
    except Exception:
        return {}


def featurize_sequence(seq):
    feat = {}
    feat.update(amino_acid_composition(seq))
    feat.update(dipeptide_composition(seq))
    feat.update(physicochemical_features(seq))
    feat.update(ctd_features(seq))
    feat.update(esm_embedding_features(seq))
    return feat


NON_FEATURE_COLUMNS = [
    "record_id", "sequence", "venom_group", "source_database",
    "source_category", "target_class", "toxicity_status", "sequence_cluster"
]


def extract_features(df):
    rows = []
    for _, row in df.iterrows():
        seq = row["sequence"]
        feat = featurize_sequence(seq)
        feat["record_id"] = row["record_id"]
        feat["sequence"] = seq
        feat["venom_group"] = row["venom_group"]
        feat["source_database"] = row["source_database"]
        feat["source_category"] = row.get("source_category", "toxin_positive")
        feat["target_class"] = row["target_class"]
        feat["toxicity_status"] = row["toxicity_status"]
        feat["sequence_cluster"] = row["sequence_cluster"]
        rows.append(feat)
    return pd.DataFrame(rows)


# ============================================================
# SECTION 8 — MODELS: pipelines, tuning, shared train/eval loop
# ============================================================
def build_feature_matrix(df, target_column, drop_extra=None):
    drop = set(NON_FEATURE_COLUMNS) | {target_column}
    if drop_extra:
        drop.update(drop_extra)
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=np.number)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def make_model_pipelines(n_classes, n_features):
    """Random Forest, SVM, XGBoost, and a Neural Network (MLP) -- the
    last one closes the gap between the abstract (which promises
    neural networks) and the original code (which had none). Every
    pipeline includes SelectKBest so no classifier sees the full
    feature matrix directly against classes with as few as 15-26
    samples."""
    k = min(config.MAX_SELECTED_FEATURES, n_features)

    pipelines = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("classifier", RandomForestClassifier(class_weight="balanced",
                                                    random_state=config.RANDOM_STATE, n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced",
                                random_state=config.RANDOM_STATE)),
        ]),
        "Neural Network": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(random_state=config.RANDOM_STATE, max_iter=800,
                                          early_stopping=True)),
        ]),
    }

    param_distributions = {
        "Random Forest": {
            "classifier__n_estimators": [300, 500, 700, 900],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        },
        "SVM": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": ["scale", "auto", 0.01, 0.1],
        },
        "Neural Network": {
            "classifier__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__learning_rate_init": [0.001, 0.01],
        },
    }

    if n_classes == 2:
        xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             random_state=config.RANDOM_STATE, n_jobs=-1)
    else:
        xgb = XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss",
                             random_state=config.RANDOM_STATE, n_jobs=-1)

    pipelines["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("classifier", xgb),
    ])
    param_distributions["XGBoost"] = {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.85, 1.0],
        "classifier__colsample_bytree": [0.7, 0.85, 1.0],
    }

    return pipelines, param_distributions


def tune_model(name, pipeline, param_dist, X_train, y_train, groups_train):
    """Leakage-aware hyperparameter search: the inner CV is grouped by
    sequence-similarity cluster, exactly like the outer split."""
    n_unique_groups = pd.Series(groups_train).nunique()
    if n_unique_groups >= config.RANDOM_SEARCH_INNER_SPLITS:
        inner_cv = StratifiedGroupKFold(n_splits=min(config.RANDOM_SEARCH_INNER_SPLITS, n_unique_groups),
                                         shuffle=True, random_state=config.RANDOM_STATE)
        fit_kwargs = {"groups": groups_train}
    else:
        inner_cv = StratifiedKFold(n_splits=max(2, min(config.RANDOM_SEARCH_INNER_SPLITS, pd.Series(y_train).value_counts().min())),
                                    shuffle=True, random_state=config.RANDOM_STATE)
        fit_kwargs = {}

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=config.RANDOM_SEARCH_ITER,
                                 cv=inner_cv, scoring="f1_macro", random_state=config.RANDOM_STATE,
                                 n_jobs=-1, refit=True, error_score=0.0)
    search.fit(X_train, y_train, **fit_kwargs)
    print(f"  {name}: best params -> {search.best_params_}")
    return search.best_estimator_


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def choose_group_split(X, y, groups, test_size=0.20):
    unique_groups = pd.Series(groups).nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise RuntimeError("Not enough sequence-similarity groups for leakage-aware split.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    candidates = list(cv.split(X, y, groups))
    return min(candidates, key=lambda pair: abs(len(pair[1]) / len(X) - test_size))


MAJORITY_BASELINE_NAME = "Majority baseline"
SIMPLE_BASELINE_NAME = "Simple baseline (length + cysteine only)"
BASELINE_MODEL_NAMES = {MAJORITY_BASELINE_NAME, SIMPLE_BASELINE_NAME}


def best_real_model_row(results_df):
    """Best-scoring row among the real models (baselines excluded)."""
    real = results_df[~results_df["model"].isin(BASELINE_MODEL_NAMES)]
    return real.sort_values("balanced_accuracy", ascending=False).iloc[0]


def matched_toxicity_subset(df):
    """v3.1: a HARD version of the toxicity task.

    Toxic and Non_toxic examples are sampled so that every
    (length bin x cysteine-fraction bin) cell holds the same number of
    each class. After this, length and cysteine content carry no
    information about the label, so a model that still scores well must
    be using something else."""
    d = df.copy()
    d["_len_bin"] = pd.cut(d["length"], bins=config.MATCH_LENGTH_BIN_EDGES, labels=False, include_lowest=True)
    d["_cys_bin"] = pd.cut(d["cysteine_fraction"], bins=config.MATCH_CYS_BIN_EDGES, labels=False, include_lowest=True)

    keep = []
    for _, cell in d.groupby(["_len_bin", "_cys_bin"]):
        tox = cell[cell["toxicity_status"] == "Toxic"]
        non = cell[cell["toxicity_status"] == "Non_toxic"]
        n = min(len(tox), len(non))
        if n == 0:
            continue
        keep.append(tox.sample(n, random_state=config.RANDOM_STATE))
        keep.append(non.sample(n, random_state=config.RANDOM_STATE))

    if not keep:
        return d.iloc[0:0].drop(columns=["_len_bin", "_cys_bin"])
    return pd.concat(keep).drop(columns=["_len_bin", "_cys_bin"]).reset_index(drop=True)


def toxicity_reliability(full_results_df, matched_results_df):
    """v3.1: the toxicity model is only called reliable if ALL hold:
      1. best model reaches the minimum balanced accuracy,
      2. it beats the length+cysteine-only baseline by a real margin,
      3. it ALSO reaches the minimum on the matched (hard) test.
    If the matched test could not be run, reliability is NOT granted.
    Returns (is_reliable, list_of_text_lines)."""
    lines = []
    full_best = best_real_model_row(full_results_df)["balanced_accuracy"]
    ok = full_best >= config.MIN_RELIABLE_BALANCED_ACCURACY
    lines.append(f"full test best balanced accuracy      : {full_best:.3f} (needs >= {config.MIN_RELIABLE_BALANCED_ACCURACY})")

    base = full_results_df[full_results_df["model"] == SIMPLE_BASELINE_NAME]
    if not base.empty:
        base_score = float(base["balanced_accuracy"].iloc[0])
        gain = full_best - base_score
        ok = ok and gain >= config.MIN_GAIN_OVER_SIMPLE_BASELINE
        lines.append(f"length+cysteine baseline              : {base_score:.3f} -> gain {gain:+.3f} (needs >= {config.MIN_GAIN_OVER_SIMPLE_BASELINE})")
    else:
        ok = False
        lines.append("length+cysteine baseline              : not available -> reliability not granted")

    if matched_results_df is None:
        ok = False
        lines.append("matched (hard) test                   : could not be run -> reliability not granted")
    else:
        m_best = best_real_model_row(matched_results_df)["balanced_accuracy"]
        ok = ok and m_best >= config.MIN_RELIABLE_BALANCED_ACCURACY
        lines.append(f"matched (hard) test best balanced acc : {m_best:.3f} (needs >= {config.MIN_RELIABLE_BALANCED_ACCURACY})")
    return bool(ok), lines


def run_classification_task(task_name, data, label_column, feature_drop_extra=None,
                            simple_baseline_columns=None):
    """Shared training/eval loop for target AND toxicity tasks.
    simple_baseline_columns: optional list of feature names; if given, a
    tiny logistic-regression model using ONLY those columns is scored on
    the same split, to show how much of the result trivial features explain."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[label_column])

    X = build_feature_matrix(data, label_column, drop_extra=feature_drop_extra)
    feature_columns = list(X.columns)
    groups = data["sequence_cluster"].values

    train_idx, test_idx = choose_group_split(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    print(f"\n[{task_name}] Leakage-aware split: {len(X_train)} train / {len(X_test)} test")

    n_classes = len(label_encoder.classes_)
    pipelines, param_dists = make_model_pipelines(n_classes, X.shape[1])

    baseline = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_predictions(y_test, baseline.predict(X_test))
    baseline_metrics["model"] = MAJORITY_BASELINE_NAME
    results = [baseline_metrics]

    if simple_baseline_columns and all(c in X.columns for c in simple_baseline_columns):
        simple = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000,
                                              random_state=config.RANDOM_STATE)),
        ])
        simple.fit(X_train[simple_baseline_columns], y_train)
        simple_metrics = evaluate_predictions(y_test, simple.predict(X_test[simple_baseline_columns]))
        simple_metrics["model"] = SIMPLE_BASELINE_NAME
        results.append(simple_metrics)
        print(f"[{task_name}] {SIMPLE_BASELINE_NAME}: balanced accuracy = {simple_metrics['balanced_accuracy']:.3f}")

    trained_models = {}
    for name, pipeline in pipelines.items():
        print(f"\n{'=' * 70}\n{task_name.upper()} MODEL: {name}\n{'=' * 70}")
        try:
            best_model = tune_model(name, pipeline, param_dists[name], X_train, y_train, groups_train)
        except Exception as exc:
            print(f"  Tuning failed for {name} ({exc}); fitting with defaults instead.")
            best_model = pipeline.fit(X_train, y_train)

        pred = best_model.predict(X_test)
        metrics = evaluate_predictions(y_test, pred)
        metrics["model"] = name
        results.append(metrics)
        trained_models[name] = best_model

        print(pd.Series(metrics))
        print(classification_report(y_test, pred, target_names=label_encoder.classes_, zero_division=0))

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(9, 7))
        plt.imshow(cm)
        plt.title(f"{name} — {task_name} Prediction")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.xticks(range(n_classes), label_encoder.classes_, rotation=45, ha="right")
        plt.yticks(range(n_classes), label_encoder.classes_)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout()
        safe_name = name.replace(" ", "_")
        plt.savefig(os.path.join(config.FIGURE_DIR, f"{task_name}_confusion_{safe_name}.png"),
                    dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config.RESULT_DIR, f"{task_name}_model_comparison.csv"), index=False)
    return trained_models, label_encoder, feature_columns, X_test, y_test, results_df


# ============================================================
# SECTION 9 — SHAP EXPLAINABILITY FOR ALL FOUR MODELS
# Random Forest / XGBoost (tree models) get exact TreeExplainer SHAP.
# SVM / Neural Network have no tree structure, so they fall back to a
# generic explainer run on a capped background + capped explained
# sample (much slower per-sample) -- a documented approximation, not
# a silent shortcut.
# ============================================================
TREE_MODEL_TYPES = (RandomForestClassifier, XGBClassifier)


def _pipeline_to_model_space(pipeline, X_reference, feature_names):
    imputer = pipeline.named_steps["imputer"]
    X_imp = imputer.transform(X_reference)

    selector = pipeline.named_steps.get("select")
    if selector is not None:
        X_sel = selector.transform(X_imp)
        selected_names = np.array(feature_names)[selector.get_support()]
    else:
        X_sel = X_imp
        selected_names = np.array(feature_names)

    scaler = pipeline.named_steps.get("scaler")
    if scaler is not None:
        X_sel = scaler.transform(X_sel)

    return pd.DataFrame(X_sel, columns=selected_names, index=X_reference.index), selected_names


def _mean_abs_shap(values):
    if values.ndim == 3:
        return np.abs(values).mean(axis=(0, 2))
    return np.abs(values).mean(axis=0)


def _save_global_summary(shap_df, output_prefix, title_suffix):
    shap_df.to_csv(os.path.join(config.RESULT_DIR, f"{output_prefix}_shap_global_importance.csv"), index=False)
    plt.figure(figsize=(10, 8))
    top = shap_df.head(20).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.title(f"Global SHAP Importance — {title_suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURE_DIR, f"{output_prefix}_shap_global.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def explain_model(model_name, pipeline, X_reference, feature_names, output_prefix):
    """Explains a single trained pipeline. Never raises -- a failed
    explanation for one model shouldn't take down the rest of the run."""
    try:
        X_model_space, selected_names = _pipeline_to_model_space(pipeline, X_reference, feature_names)
        classifier = pipeline.named_steps["classifier"]

        if isinstance(classifier, TREE_MODEL_TYPES):
            explainer = shap.TreeExplainer(classifier)
            explanation = explainer(X_model_space)
        else:
            rng = np.random.default_rng(config.RANDOM_STATE)
            n = len(X_model_space)
            bg_size = min(config.SHAP_BACKGROUND_SIZE, n)
            bg_idx = rng.choice(n, size=bg_size, replace=False)
            background = X_model_space.iloc[bg_idx]
            explain_size = min(config.SHAP_EXPLAIN_SAMPLE_SIZE, n)
            explain_idx = rng.choice(n, size=explain_size, replace=False)
            sample = X_model_space.iloc[explain_idx]
            explainer = shap.Explainer(classifier.predict_proba, background, seed=config.RANDOM_STATE)
            explanation = explainer(sample)

        mean_abs = _mean_abs_shap(explanation.values)
        shap_df = pd.DataFrame({"feature": selected_names, "mean_abs_shap": mean_abs}) \
            .sort_values("mean_abs_shap", ascending=False)

        prefix = f"{output_prefix}_{model_name.replace(' ', '_').lower()}"
        _save_global_summary(shap_df, prefix, f"{output_prefix} — {model_name}")

        return {"explainer": explainer, "explanation": explanation,
                "shap_df": shap_df, "selected_features": selected_names}
    except Exception as exc:
        print(f"SHAP warning ({model_name}, {output_prefix}): {exc}")
        return {"explainer": None, "explanation": None, "shap_df": None, "selected_features": None}


def explain_all_models(trained_models, X_reference, feature_names, output_prefix):
    results = {}
    for name, pipeline in trained_models.items():
        kind = "exact TreeExplainer" if isinstance(pipeline.named_steps["classifier"], TREE_MODEL_TYPES) else "approximate, sampled"
        print(f"Running SHAP for {output_prefix} / {name} ({kind})...")
        results[name] = explain_model(name, pipeline, X_reference, feature_names, output_prefix)
    return results


# ============================================================
# SECTION 10 — REAL PATHWAY CROSS-REFERENCES (UniProt lookup)
# NOT a prediction -- a plain database lookup, labeled as such
# everywhere. Kept separate from the static descriptive-text table.
# ============================================================
TARGET_BROAD_CONTEXT = {
    "Voltage-gated sodium channel": "Electrical signaling / action potential regulation",
    "Potassium channel": "Membrane potential and electrical signaling",
    "Calcium channel": "Calcium signaling / excitation-contraction or secretion",
    "Nicotinic acetylcholine receptor": "Cholinergic neurotransmission / neuromuscular signaling",
    "Glutamate receptor": "Glutamatergic neurotransmission / synaptic signaling",
    "G-protein-coupled receptor": "GPCR-mediated signal transduction",
    "Enzyme": "Enzymatic or biochemical process; specific pathway requires target ID",
    "Membrane-associated target": "Cell-surface or membrane-associated biological process",
}


def build_pathway_crossreferences(master):
    rows = []
    for _, row in master[master["source_category"] == "toxin_positive"].iterrows():
        kegg = row.get("xref_kegg")
        reactome = row.get("xref_reactome")
        kegg_empty = pd.isna(kegg) or str(kegg).strip() in ("", "nan")
        reactome_empty = pd.isna(reactome) or str(reactome).strip() in ("", "nan")
        if kegg_empty and reactome_empty:
            continue
        rows.append({"record_id": row["record_id"], "target_class": row["target_class"],
                      "kegg_ids": kegg, "reactome_ids": reactome,
                      "lookup_method": "UniProt cross-reference (database lookup, not a model prediction)"})

    pathway_df = pd.DataFrame(rows)
    pathway_path = os.path.join(config.RESULT_DIR, "pathway_crossreferences_uniprot.csv")
    pathway_df.to_csv(pathway_path, index=False)

    n_positive = max(1, (master["source_category"] == "toxin_positive").sum())
    coverage = len(pathway_df) / n_positive
    print(f"\nPathway cross-reference coverage: {coverage:.1%} of toxin-positive records "
          f"have a KEGG or Reactome ID in UniProt.")
    return pathway_df, pathway_path, coverage


def save_broad_context_table():
    context_rows = [
        {"target_class": t, "broad_descriptive_context_NOT_a_pathway": c,
         "real_pathway_data": "See pathway_crossreferences_uniprot.csv instead"}
        for t, c in TARGET_BROAD_CONTEXT.items()
    ]
    path = os.path.join(config.RESULT_DIR, "target_class_broad_context_static_text.csv")
    pd.DataFrame(context_rows).to_csv(path, index=False)
    return path


# ============================================================
# SECTION 11 — PREDICTION / PRIORITIZATION FOR NEW PEPTIDES
# ============================================================
def sequence_to_feature_row(sequence):
    seq = clean_sequence(sequence)
    if seq is None:
        raise ValueError("Invalid peptide sequence. Use only the 20 standard amino acids.")
    return pd.DataFrame([featurize_sequence(seq)])


def calculate_priority_score(target_confidence, toxicity_prediction, toxicity_confidence,
                              toxicity_model_reliable=True, novelty_score=0.5):
    """Transparent research-triage score, NOT a clinical/toxicological
    risk score. If the toxicity model didn't clear the reliability
    bar, its contribution is neutralized (fixed at 0.5)."""
    target_confidence = float(target_confidence or 0)
    toxicity_confidence = float(toxicity_confidence or 0)

    if not toxicity_model_reliable:
        safety_component = 0.5
    elif toxicity_prediction == "Non_toxic":
        safety_component = toxicity_confidence
    elif toxicity_prediction == "Toxic":
        safety_component = 1.0 - toxicity_confidence
    else:
        safety_component = 0.5

    return 0.50 * target_confidence + 0.30 * safety_component + 0.20 * float(novelty_score)


class VenomAIPredictor:
    """Bundles everything needed to score a new peptide sequence so
    prediction doesn't require re-running data collection or training.
    Save with joblib.dump(predictor, path); reload with joblib.load."""

    def __init__(self):
        self.calibrated_target_model = None
        self.calibrated_toxicity_model = None
        self.trained_target_models = {}
        self.trained_toxicity_models = {}
        self.target_label_encoder = None
        self.toxicity_label_encoder = None
        self.target_feature_columns = []
        self.toxicity_feature_columns = []
        self.target_reliable = False
        self.toxicity_reliable = False

    def predict_one(self, sequence):
        seq = clean_sequence(sequence)
        X_new = sequence_to_feature_row(seq)

        report = {
            "sequence": seq, "length": len(seq),
            "target_prediction": None, "target_confidence": None,
            "target_model_reliable": self.target_reliable,
            "toxicity_prediction": None, "toxicity_confidence": None,
            "toxicity_model_reliable": self.toxicity_reliable,
        }

        if self.calibrated_target_model is not None:
            X_t = X_new.reindex(columns=self.target_feature_columns, fill_value=0)
            probs = self.calibrated_target_model.predict_proba(X_t)[0]
            idx = int(np.argmax(probs))
            report["target_prediction"] = self.target_label_encoder.inverse_transform([idx])[0]
            report["target_confidence"] = float(probs[idx])
            report["target_probabilities"] = pd.DataFrame({
                "target_class": self.target_label_encoder.classes_, "probability": probs
            }).sort_values("probability", ascending=False)

        if self.calibrated_toxicity_model is not None:
            X_tox = X_new.reindex(columns=self.toxicity_feature_columns, fill_value=0)
            probs = self.calibrated_toxicity_model.predict_proba(X_tox)[0]
            idx = int(np.argmax(probs))
            report["toxicity_prediction"] = self.toxicity_label_encoder.inverse_transform([idx])[0]
            report["toxicity_confidence"] = float(probs[idx])
            report["toxicity_probabilities"] = pd.DataFrame({
                "toxicity_class": self.toxicity_label_encoder.classes_, "probability": probs
            }).sort_values("probability", ascending=False)

        return report

    def explain_one(self, sequence, task="target", model_name="Random Forest"):
        """Per-peptide SHAP explanation for ANY of the four trained
        models on either task -- not just Random Forest."""
        models = self.trained_target_models if task == "target" else self.trained_toxicity_models
        feature_columns = self.target_feature_columns if task == "target" else self.toxicity_feature_columns
        if model_name not in models:
            return None
        X_new = sequence_to_feature_row(sequence).reindex(columns=feature_columns, fill_value=0)
        result = explain_model(model_name, models[model_name], X_new, feature_columns,
                                output_prefix=f"single_peptide_{task}")
        return result["shap_df"]

    def predict_and_rank(self, sequences):
        rows = []
        for seq in sequences:
            try:
                report = self.predict_one(seq)
                score = calculate_priority_score(
                    report["target_confidence"], report["toxicity_prediction"],
                    report["toxicity_confidence"], report["toxicity_model_reliable"],
                )
                rows.append({
                    "sequence": report["sequence"], "length": report["length"],
                    "predicted_target": report["target_prediction"],
                    "target_confidence": report["target_confidence"],
                    "target_model_reliable": report["target_model_reliable"],
                    "predicted_toxicity": report["toxicity_prediction"],
                    "toxicity_confidence": report["toxicity_confidence"],
                    "toxicity_model_reliable": report["toxicity_model_reliable"],
                    "priority_score": score,
                })
            except Exception as exc:
                rows.append({"sequence": seq, "error": str(exc)})
        return pd.DataFrame(rows).sort_values("priority_score", ascending=False, na_position="last")


# ============================================================
# SECTION 12 — ORCHESTRATION: run the whole pipeline
# ============================================================
def quality_control(master):
    master["sequence"] = master["sequence"].apply(clean_sequence)
    master = master[master["sequence"].notna()].copy()
    master["sequence_hash"] = master["sequence"].apply(sequence_hash)
    master["sequence_length"] = master["sequence"].str.len()
    master = master.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)

    def detect_fragment(row):
        text = " ".join(
            str(row.get(c, "")) for c in ["protein_name", "function_annotation", "fragment", "record_id", "header"]
            if not pd.isna(row.get(c, np.nan))
        ).lower()
        terms = ["fragment", "partial", "truncated", "n-terminal fragment", "c-terminal fragment"]
        return any(term in text for term in terms)

    master["fragment_like"] = master.apply(detect_fragment, axis=1)
    return master


def save_bar(series, title, filename, rotation=30):
    plt.figure(figsize=(11, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Number of records")
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURE_DIR, filename), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def run_confound_check(target_data):
    """Does target classification still work WITHIN a single venom
    group? Low performance here means the full-dataset result is
    partly explained by organism/database identity, not target biology."""
    print("\n" + "#" * 70)
    print("CONFOUND CHECK: target classification WITHIN a single venom group")
    print("(low performance here means the full-dataset result is partly")
    print(" explained by organism/database identity, not target biology)")
    print("#" * 70)

    for group_name in target_data["venom_group"].unique():
        subset = target_data[target_data["venom_group"] == group_name]
        sub_counts = subset["target_class"].value_counts()
        sub_valid = sub_counts[sub_counts >= config.MIN_TARGET_CLASS_SIZE].index
        subset = subset[subset["target_class"].isin(sub_valid)]
        if subset["target_class"].nunique() < 2:
            print(f"  {group_name}: not enough within-group classes to test.")
            continue
        try:
            le = LabelEncoder()
            y_sub = le.fit_transform(subset["target_class"])
            X_sub = build_feature_matrix(subset, "target_class")
            groups_sub = subset["sequence_cluster"].values
            tr_idx, te_idx = choose_group_split(X_sub, y_sub, groups_sub)
            rf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=min(config.MAX_SELECTED_FEATURES, X_sub.shape[1]))),
                ("classifier", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                        random_state=config.RANDOM_STATE, n_jobs=-1)),
            ])
            rf.fit(X_sub.iloc[tr_idx], y_sub[tr_idx])
            pred_sub = rf.predict(X_sub.iloc[te_idx])
            m = evaluate_predictions(y_sub[te_idx], pred_sub)
            print(f"  {group_name}: balanced_accuracy={m['balanced_accuracy']:.3f}, "
                  f"f1_macro={m['f1_macro']:.3f}, n_test={len(te_idx)}")
        except Exception as exc:
            print(f"  {group_name}: confound check failed ({exc})")
    print("#" * 70)


def calibrate(pipeline, X_train, y_train):
    try:
        calibrated = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        return calibrated
    except Exception as exc:
        print("Calibration failed, falling back to uncalibrated model:", exc)
        return pipeline


def run_pipeline():
    # ---- 1. Collect + standardize + QC ----
    datasets = [standardize(df) for df in collect_data()]
    master = pd.concat(datasets, ignore_index=True)
    master = quality_control(master)

    # ---- 2. Label (target + toxicity, with real negative controls) ----
    master = label_master(master)

    # ---- 3. Leakage-aware clustering ----
    groups, clustering_method, clustering_is_strict = sequence_similarity_groups(master)
    master["sequence_cluster"] = groups
    if not clustering_is_strict:
        print(">>> Leakage control ran on the APPROXIMATE fallback, not CD-HIT. <<<")

    # ---- 3b. Clean the "non-toxic" controls (v3.1) ----
    master, control_cleaning_counts = flag_contaminated_controls(master)

    # ---- 4. Features ----
    features_df = extract_features(master)
    numeric_columns = [c for c in features_df.columns if c not in NON_FEATURE_COLUMNS]
    features_df[numeric_columns] = features_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    features_path = os.path.join(config.FEATURE_DIR, "VenomAI_v3_features.csv")
    features_df.to_csv(features_path, index=False)

    # ---- 5. EDA ----
    save_bar(master["venom_group"].value_counts(), "VenomAI Records by Venom Group", "venom_group_distribution.png")
    save_bar(master["target_class"].value_counts(), "Target Annotation Distribution", "target_distribution.png")
    save_bar(master["toxicity_status"].value_counts(), "Toxicity Evidence Distribution (with real negatives)", "toxicity_distribution.png")

    predictor = VenomAIPredictor()

    # ---- 6. Target classification ----
    target_data = features_df[features_df["target_class"] != "Unknown"].copy()
    target_data = target_data[target_data["target_class"] != "Not_applicable"].copy()
    target_counts = target_data["target_class"].value_counts()
    valid_target_classes = target_counts[target_counts >= config.MIN_TARGET_CLASS_SIZE].index
    target_data = target_data[target_data["target_class"].isin(valid_target_classes)].copy()

    print("\nTARGET DATASET")
    print(target_data["target_class"].value_counts())

    X_target_test = None
    target_feature_columns = []
    if target_data["target_class"].nunique() >= 2:
        (trained_target_models, target_label_encoder, target_feature_columns,
         X_target_test, y_target_test, target_results_df) = run_classification_task(
            "target", target_data, "target_class"
        )
        predictor.trained_target_models = trained_target_models
        predictor.target_label_encoder = target_label_encoder
        predictor.target_feature_columns = target_feature_columns

        for name, model in trained_target_models.items():
            joblib.dump(model, os.path.join(config.MODEL_DIR, f"target_{name.replace(' ', '_').lower()}.joblib"))
        joblib.dump(target_label_encoder, os.path.join(config.MODEL_DIR, "target_label_encoder.joblib"))

        best_row = best_real_model_row(target_results_df)
        predictor.target_reliable = bool(best_row["balanced_accuracy"] >= config.MIN_RELIABLE_BALANCED_ACCURACY)

        run_confound_check(target_data)
    else:
        print("\nTarget model not trained: insufficient validated target classes.")

    # ---- 7. Toxicity classification (with real negative controls) ----
    toxicity_data = features_df[features_df["toxicity_status"].isin(["Toxic", "Non_toxic"])].copy()
    tox_counts = toxicity_data["toxicity_status"].value_counts()
    print("\nTOXICITY DATASET (with real negative controls)")
    print(tox_counts)

    X_tox_test = None
    toxicity_feature_columns = []
    toxicity_results_df = None
    matched_results_df = None
    toxicity_reliability_lines = []
    if len(tox_counts) == 2 and tox_counts.min() >= config.MIN_TOXICITY_CLASS_SIZE:
        (trained_toxicity_models, toxicity_label_encoder, toxicity_feature_columns,
         X_tox_test, y_tox_test, toxicity_results_df) = run_classification_task(
            "toxicity", toxicity_data, "toxicity_status",
            simple_baseline_columns=["length", "cysteine_fraction"]
        )
        predictor.trained_toxicity_models = trained_toxicity_models
        predictor.toxicity_label_encoder = toxicity_label_encoder
        predictor.toxicity_feature_columns = toxicity_feature_columns

        for name, model in trained_toxicity_models.items():
            joblib.dump(model, os.path.join(config.MODEL_DIR, f"toxicity_{name.replace(' ', '_').lower()}.joblib"))
        joblib.dump(toxicity_label_encoder, os.path.join(config.MODEL_DIR, "toxicity_label_encoder.joblib"))

        # ---- 7b. Matched (hard) toxicity test: length + cysteine can't help ----
        matched_data = matched_toxicity_subset(toxicity_data)
        matched_counts = matched_data["toxicity_status"].value_counts() if len(matched_data) else pd.Series(dtype=int)
        print("\nMATCHED (HARD) TOXICITY TEST: classes balanced within every length x cysteine cell")
        print(matched_counts)
        if len(matched_counts) == 2 and matched_counts.min() >= config.MIN_TOXICITY_CLASS_SIZE:
            _, _, _, _, _, matched_results_df = run_classification_task(
                "toxicity_matched", matched_data, "toxicity_status",
                simple_baseline_columns=["length", "cysteine_fraction"]
            )
        else:
            print("Matched test could NOT be run: too few examples remain after matching.")

        predictor.toxicity_reliable, toxicity_reliability_lines = toxicity_reliability(
            toxicity_results_df, matched_results_df
        )
        print("\nTOXICITY RELIABILITY CHECK")
        for line in toxicity_reliability_lines:
            print("  " + line)
        print(f"  -> toxicity model reliable? {predictor.toxicity_reliable}")
        if not predictor.toxicity_reliable:
            print(">>> WARNING: toxicity predictions are flagged UNRELIABLE; the priority score")
            print(">>> ignores them (fixed at 0.5) until the checks above pass.")
    else:
        print("\nToxicity model NOT trained: even with real negative controls, "
              f"one class still has fewer than {config.MIN_TOXICITY_CLASS_SIZE} examples.")

    # ---- 8. Probability calibration ----
    if "Random Forest" in predictor.trained_target_models:
        X_full = build_feature_matrix(target_data, "target_class")
        y_full = predictor.target_label_encoder.transform(target_data["target_class"])
        predictor.calibrated_target_model = calibrate(predictor.trained_target_models["Random Forest"], X_full, y_full)

    if "Random Forest" in predictor.trained_toxicity_models:
        X_full = build_feature_matrix(toxicity_data, "toxicity_status")
        y_full = predictor.toxicity_label_encoder.transform(toxicity_data["toxicity_status"])
        predictor.calibrated_toxicity_model = calibrate(predictor.trained_toxicity_models["Random Forest"], X_full, y_full)

    # ---- 9. SHAP for ALL models ----
    if predictor.trained_target_models and X_target_test is not None:
        explain_all_models(predictor.trained_target_models, X_target_test, target_feature_columns, "target")
    if predictor.trained_toxicity_models and X_tox_test is not None:
        explain_all_models(predictor.trained_toxicity_models, X_tox_test, toxicity_feature_columns, "toxicity")

    # ---- 10. Real pathway cross-references + static context table ----
    _, pathway_path, pathway_coverage = build_pathway_crossreferences(master)
    save_broad_context_table()

    # ---- 11. Save datasets + predictor + metadata ----
    master_path = os.path.join(config.PROCESSED_DIR, "VenomAI_v3_master_dataset.csv")
    master.to_csv(master_path, index=False)

    predictor_path = os.path.join(config.MODEL_DIR, "venomai_predictor.joblib")
    joblib.dump(predictor, predictor_path)

    metadata = {
        "project": "VenomAI", "version": "3.2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": int(len(master)), "unique_sequences": int(master["sequence"].nunique()),
        "target_classes": master["target_class"].value_counts().to_dict(),
        "toxicity_status": master["toxicity_status"].value_counts().to_dict(),
        "venom_groups": master["venom_group"].value_counts().to_dict(),
        "clustering_method": clustering_method, "clustering_is_strict_cdhit": clustering_is_strict,
        "feature_set": "AAC + dipeptide + physicochemical + CTD" + (" + ESM" if config.USE_ESM_EMBEDDINGS else ""),
        "shap_coverage": "all trained models (tree-exact for RF/XGBoost, sampled-approximate for SVM/Neural Network)",
        "pathway_crossreference_coverage": pathway_coverage,
        "target_model_reliable": predictor.target_reliable,
        "toxicity_model_reliable": predictor.toxicity_reliable,
        "reliability_threshold_balanced_accuracy": config.MIN_RELIABLE_BALANCED_ACCURACY,
        "control_cleaning": control_cleaning_counts,
        "conoserver_filter": CONOSERVER_FILTER_SUMMARY,
        "target_label_sources": master["target_label_source"].value_counts().to_dict(),
        "toxicity_full_test": (toxicity_results_df.to_dict(orient="records") if toxicity_results_df is not None else None),
        "toxicity_matched_test": (matched_results_df.to_dict(orient="records") if matched_results_df is not None else None),
        "toxicity_reliability_checks": toxicity_reliability_lines,
    }
    with open(os.path.join(config.LOG_DIR, "VenomAI_v3_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=4)

    print("\n" + "=" * 78)
    print("VENOMAI v3.2 FINISHED")
    print("=" * 78)
    print("Master dataset    :", master_path)
    print("Feature matrix    :", features_path)
    print("Pathway lookup    :", pathway_path)
    print("Predictor bundle  :", predictor_path)
    print("Target reliable?  :", predictor.target_reliable)
    print("Toxicity reliable?:", predictor.toxicity_reliable)
    print("\nSCIENTIFIC LIMITATIONS THAT REMAIN (kept, not hidden):")
    print("1. Target labels are curated from database TEXT, not independent")
    print("   experimental ground truth.")
    print("2. Toxicity negative controls are non-toxin-KEYWORD proteins that were")
    print("   cleaned by venom-like NAME and by closeness to known toxins, but they")
    print("   are still not experimentally confirmed non-toxic peptides, and the")
    print("   name filter will miss some.")
    print("3. Taxonomic coverage is limited to spiders, scorpions, cone snails.")
    print("4. Calibration uses ordinary (non-grouped) CV internally.")
    print("5. Pathway coverage depends on existing UniProt KEGG/Reactome xrefs.")
    print("6. Predictions require experimental validation before any")
    print("   therapeutic or biological claim is made from them.")
    if not clustering_is_strict:
        print("7. CD-HIT was NOT available this run -- leakage control used the")
        print("   approximate fallback. Re-run where CD-HIT installs cleanly")
        print("   before treating these numbers as final.")
    print("=" * 78)

    return predictor, metadata


# ============================================================
# SECTION 13 — RUN IT
# ============================================================
if __name__ == "__main__":
    predictor, metadata = run_pipeline()


# ============================================================
# ONE-OFF PREDICTION EXAMPLE
# After the cell above finishes, you can score any new sequence like
# this in a NEW Colab cell (predictor is already in memory, or reload
# it later with: predictor = joblib.load(
#   "/content/VenomAI_v3/models/venomai_predictor.joblib")
# ============================================================
# report = predictor.predict_one("GCKEFGWPCNGEACCS")
# print(report["target_prediction"], report["target_confidence"])
# print(report["toxicity_prediction"], report["toxicity_model_reliable"])
#
# ranked = predictor.predict_and_rank(["SEQ1...", "SEQ2...", "SEQ3..."])
# print(ranked)
#
# shap_for_svm = predictor.explain_one("GCKEFGWPCNGEACCS", task="target", model_name="SVM")
# print(shap_for_svm.head(10))