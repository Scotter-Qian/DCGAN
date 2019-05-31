from os import listdir
from os.path import join
import pickle
import numpy as np
import cv2
import os


def search(path):
    filepath_set = []
    a = 0
    for name in os.listdir(path):
        name_path = path + os.sep + name
        if os.path.isfile(name_path):
            if name_path.endswith(".jpg"):
                filepath_set.append(name_path)
                a+=1
    return filepath_set, a

def saveData(x, y, filepath):
    file = open(filepath, 'wb')
    assert np.all(np.logical_or(y==0, y==1))
    pickle.dump({"x":x, "y":y}, file)
    file.close()

if __name__=="__main__":
    directory = r"D:\data"
    labels = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    s =listdir(directory)
    print(len(s), len(labels))

    for i in range(len(s)):
        path = join(directory, s[i])
        y = labels[i]
        #print(path)
        filepath_set, a = search(path)
        for j in range(a):
            x = cv2.imread(filepath_set[j], cv2.IMREAD_GRAYSCALE)
            print(filepath_set[j])
            saveData(x, y, filepath_set[j])
            portion  = os.path.splitext(filepath_set[j])
            if portion[1] == ".jpg":
                newname = portion[0]  +  ".xy"
                os.rename(filepath_set[j], newname)



