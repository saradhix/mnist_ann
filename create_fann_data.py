import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import sys

print "Loading MNIST data"
mnist = fetch_mldata("MNIST Original")
indices = arange(len(mnist.data))
n_train = 60000
n_test = 10000
train_idx = arange(0,n_train)
test_idx = arange(n_train,n_train+n_test)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
X_train = X_train / 255.
X_test = X_test / 255.

pca = PCA(n_components=50)
print "Fitting and transforming training data"
X_train_transformed = pca.fit_transform(X_train)
print "Transforming test data"
X_test_transformed = pca.transform(X_test)

#create the y_vector
n = 10 #10 labels 1 to 10 both inclusive
y_vector = [['1' if i == j else '0' for i in range(n)] for j in range(n)]
#print y_vector
#print X_transformed.shape

#print X_transformed[0].tolist()

#start writing to stdout in the format of fann input file
#The first line consists of three numbers: The first is the number of training pairs in the file, the second is the number of inputs and the third is the number of outputs. The rest of the file is the actual training data, consisting of one line with inputs, one with outputs etc.
print "Creating training file"
fp = open('mnist_train.data','w')
line ="%d %d %d" % ( X_train_transformed.shape[0], len(X_train_transformed[0]),10)
fp.write(line)
fp.write('\n')
for i, label in enumerate(y_train):
  fp.write(' '.join(list(map(str,X_train_transformed[i].tolist()))))
  fp.write('\n')
  fp.write(' '.join(y_vector[int(label)]))
  fp.write('\n')
fp.close()
print "Creating test file"
fp = open('mnist_test.data','w')
line ="%d %d %d" % ( X_test_transformed.shape[0], len(X_test_transformed[0]),10)
fp.write(line)
fp.write('\n')
for i, label in enumerate(y_test):
  fp.write(' '.join(list(map(str,X_test_transformed[i].tolist()))))
  fp.write('\n')
  fp.write(' '.join(y_vector[int(label)]))
  fp.write('\n')
fp.close()
print "Done"
    
