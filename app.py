import joblib
from flask import Flask, render_template, request, jsonify
from venomai_v3 import *   # keep this line: it lets joblib find VenomAIPredictor

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
predictor = joblib.load("model/venomai_predictor_web.joblib")   # loads once, at startup

VALID = set("ACDEFGHIKLMNPQRSTVWY")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tool")
def tool():
    return render_template("tool.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    seq = (data.get("sequence") or "").strip().upper().replace(" ", "")

    if not seq or not set(seq) <= VALID:
        return jsonify(error="Use only the 20 standard amino-acid letters."), 400
    if not 5 <= len(seq) <= 200:
        return jsonify(error="Sequence must be between 5 and 200 letters."), 400

    report = predictor.predict_one(seq)
    # tables can't be sent as-is, so turn the top 3 rows into plain lists
    for key in ("target_probabilities", "toxicity_probabilities"):
        if key in report:
            report[key] = report[key].head(3).to_dict("records")
    return jsonify(report)

AA_NAMES = {"A":"alanine","C":"cysteine","D":"aspartate","E":"glutamate","F":"phenylalanine",
            "G":"glycine","H":"histidine","I":"isoleucine","K":"lysine","L":"leucine",
            "M":"methionine","N":"asparagine","P":"proline","Q":"glutamine","R":"arginine",
            "S":"serine","T":"threonine","V":"valine","W":"tryptophan","Y":"tyrosine"}

PHYSICO = {
    "length": "Peptide length",
    "molecular_weight": "Molecular weight",
    "isoelectric_point": "Isoelectric point (the pH where it has no net charge)",
    "gravy": "Water-loving vs water-avoiding character (GRAVY)",
    "aromaticity": "Share of aromatic amino acids",
    "instability_index": "Predicted instability",
    "cysteine_fraction": "Share of cysteines (the residues that form disulfide bridges)",
}

def plain_name(feature):
    """Turn a technical feature name into a sentence a person can read."""
    if feature in PHYSICO:
        return PHYSICO[feature]
    if feature.startswith("AAC_"):
        aa = feature[4:]
        return f"How much {AA_NAMES.get(aa, aa)} ({aa}) the peptide contains"
    if feature.startswith("DP_"):
        pair = feature[3:]
        return f"How often {pair[0]} is directly followed by {pair[1]}"
    if feature.startswith("CTD_"):
        return "A chemical property pattern along the chain (" + feature[4:].replace("_", " ") + ")"
    return feature.replace("_", " ")

@app.route("/why")
def why():
    return render_template("why.html")

@app.route("/explain")
def explain():
    seq = (request.args.get("seq") or "").strip().upper()
    if not seq or not set(seq) <= VALID or not 5 <= len(seq) <= 200:
        return jsonify(error="Invalid sequence."), 400

    shap_df = predictor.explain_one(seq, task="target", model_name="Random Forest")
    if shap_df is None:
        return jsonify(error="Could not explain this sequence."), 500

    top = shap_df.head(8)
    features = [{"feature": f, "label": plain_name(f), "value": float(v)}
                for f, v in zip(top["feature"], top["mean_abs_shap"])]
    return jsonify(sequence=seq, model="Random Forest", features=features)

if __name__ == "__main__":
    app.run(debug=False)
