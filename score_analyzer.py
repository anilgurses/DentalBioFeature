import os
import pandas as pd 
import numpy  as np 
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


scores = pd.read_csv("outputs/Alexnet_21-03-20:38_score.csv")
print(classification_report(scores["Real"], scores["Predicted"]))

fpr, tpr, thresholds = roc_curve(scores["Real"], scores["Predicted"])


plt.figure()

plt.plot(fpr, tpr, color='darkorange')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()