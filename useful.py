import numpy as np
from sklearn.tree import plot_tree
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import os
os.environ["XDG_SESSION_TYPE"] = "xcb"



def load_from_file(folder,set="train",categorical=True):
    x = np.loadtxt(f"{folder}/{set}_x.txt")
    y = np.loadtxt(f"{folder}/{set}_y.txt")
    if categorical == False:
        y = y.argmax(axis=1)
    return x,y


def print_scores(actual,predicted):
    print("accuracy: {}".format(accuracy_score(actual,predicted)))
    print(confusion_matrix(actual,predicted))
    precision, recall, fscore, support = score(actual, predicted)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    return 

def draw_tree(tree, feature_names,tree_name="decision_tree.png"):
    fig = plt.figure(figsize=(50,40))
    _ = plot_tree(tree,filled=True,feature_names=feature_names)
    fig.savefig(tree_name)

