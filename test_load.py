import joblib
from venomai_v3 import *   # this makes the VenomAIPredictor class findable

predictor = joblib.load("model/venomai_predictor_web.joblib")
report = predictor.predict_one("GCKEFGWPCNGEACCS")
print(report["target_prediction"], report["target_confidence"])
print(report["toxicity_prediction"], report["toxicity_model_reliable"])