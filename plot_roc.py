import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Twitter Sentiment Analysis.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',help='Input file containing (label,score) pairs used to  plot ROC')
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    y_true = data['label'].values
    y_probas = data['score'].values
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=0)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Print ROC curve
    plt.plot(fpr,tpr, label='Factorization Machine')
    plt.legend()
    plt.show() 
    # plt.savefig('fm_roc.pdf')