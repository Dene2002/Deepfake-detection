
from typing import Tuple
import pandas as pd

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

test_csv=pd.read_excel("D:\\vit btech final year 2023\\Capstone\\whisper_code\\Sample_Dataset.xlsx")
#
predicted_labels = test_csv['PROB_LCNN'].tolist()
true_labels = test_csv['tlabel'].tolist()


def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, y_score)
    # Plot ROC curve
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for LCNN Under ITW ')
    plt.legend(loc="lower right")
    plt.show()
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    print("EER value:", eer)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr
    # eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    # eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    # print(eer)
    # return eer_threshold,eer,fpr, tpr


print(calculate_eer(true_labels, predicted_labels))



