import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

#initialize desired features (these are metrics the system generates based on info from the motion sensors)
features = ['meanSignalLenght', 'amplCor', 'syncVar', 'metr6.velSum']

#we need to be able to look how each metric changes over time 
window_length = 4

col_list = []

for x in features:
    count = 1
    while count <= window_length:
        col_list.append((str(count) + x))
        count += 1

#now we have created a column list based on metrics fitting a chosen window length 

#function to populate df
def makedata(filename, label):
    global df
    df = pd.DataFrame(columns=col_list)
    with open(filename, "r") as file:
        window_count = 1
        row= 0
        previous = []
        for line in file:
            new_win = line.find('= windowEndTimer')
            if (new_win != -1):
                window_count += 1
            if window_count > window_length:
                window_count = 1
                row += 1
            for x in features:
                regexp = re.compile(x + r'.*?([0-9.-]+)')
                match = regexp.search(line)
                if match and (match.group(1)) != previous:
                    df.at[row,str(str(window_count) +x)] = match.group(1)
                    previous = match.group(1)
    df['label'] = label

#files with chosen labels
makedata("../file/hand_highway.txt", 0)
hand_highway = df.copy(deep= True)

makedata("../file/pocket_highway.txt", 1)
pocket_highway = df.copy(deep= True)

makedata("../file/home_use.txt", 2)
df = pd.concat([pocket_highway, df, hand_highway], ignore_index=True)


#setting features and labels
labels = np.array(df['label'])
features= df.drop('label', axis = 1)

features = np.array(features)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(features, labels,
                                                          stratify = labels,
                                                          test_size = 0.5,
                                                          random_state = 3)

from sklearn.ensemble import RandomForestClassifier

#we choose a model
clf_rf = RandomForestClassifier(n_estimators=100, random_state=12, class_weight='balanced')
clf_rf.fit(x_train, y_train)

actual = y_val
predictions = clf_rf.predict(x_val)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(actual,predictions))

from sklearn.metrics import classification_report
print(classification_report(actual, predictions))
