# validacao/metrics.py
from sklearn.metrics import f1_score

def calculate_f1_scores(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    return f1_macro, f1_weighted
