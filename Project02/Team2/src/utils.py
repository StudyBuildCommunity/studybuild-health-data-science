import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import numpy as np


sns.set_theme(style="whitegrid")

def draw_confuison_matrix(y_test, y_pred):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Q4: Baseline Model Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()


def tune_threshold(y_true, y_prob, min_recall=None):
    """F1-optimal threshold; with `min_recall`, the best-F1 threshold among those reaching that recall."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]                     # the last PR point has no threshold
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if min_recall is not None:
        f1 = np.where(rec >= min_recall, f1, -1)
    i = int(np.argmax(f1))
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])