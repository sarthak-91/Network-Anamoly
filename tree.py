import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from useful import load_from_file, print_scores, draw_tree
from sklearn.model_selection import train_test_split
from data_explore import columns


if __name__ == "__main__":
    x,y=load_from_file(folder="binary",set="train",categorical=False)
    model =DecisionTreeClassifier()
    model.fit(x,y)
    x,y = load_from_file(folder="binary",set="test",categorical=False)
    y_pred = model.predict(x)
    print_scores(y,y_pred)
    draw_tree(model,columns)