#This script trains the LDA, SVM-LIN, SVM-RBF, MLP, and KNN classifiers for a set of training set sizes

# https://nicholastsmith.wordpress.com/2016/02/13/wine-classification-using-linear-discriminant-analysis-with-python-and-scikit-learn/

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import numpy as np

from sklearn import cross_validation, svm, neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import random

def write2file(filename,item,truncate):
	with open(filename, 'r+') as f:
		if (truncate == 1): f.truncate()
		f.write(item)
		f.write("\n")

np.set_printoptions(precision=2)

#Read data from csv file containing the extracted features
data = np.loadtxt('features_celebs3_squared_v1.txt', delimiter=' ')
data_test = np.loadtxt('features_celebs_extra_sorted_noref.txt', delimiter=' ')

#set variables
confusion = []
SEED = 123 #for randomizing train/test split
train_p_inc = 20 #increment of train set size

#Get indices
ind = data[:, 0]

#Get the targets (first column of file)
y = data[:, 1]
y_test = data_test[:, 1]

#Remove targets from input data; features set A is the one originally used in the paper, training set B includes facial width ratios; features set C does not use angle-based features
A = data[:, 2:20]
B = data[:,2:]

A_test = data_test[:, 2:20]
B_test = data_test[:,2:]

scaler_test = StandardScaler()
scaler_test.fit(A_test)
A_test_scaled = scaler_test.transform(A_test)

#remove index 3
C1 = np.hstack( [B[:,:1], B[:,(1+1):] ] )
C = np.hstack( [C1[:,:2], C1[:,18:] ] )

# Split the data into a training set and a test set
def split_data(train_p,seed):
	index = list(range(0,100))
	random.seed(seed)
	random.shuffle(index)

	trn = index[:train_p]
	tst = index[train_p:]

	def getindices(list):
		new_list = []
		for n in range(0,len(list)):
			new_list.append(list[n])
			for i in range(1,5):
				new_list.append(list[n]+i*100)
		return new_list

	trn_all = getindices(trn)
	tst_all = getindices(tst)
	
	return trn_all, tst_all

features = A #specify which feature set to use
train_p = 0 
eval_all = 1 #1 if evaluate all data, 0 if evaluate training data only, 2 if use test_set

data_evaluated = ['train', 'all']

#out_file contains the confusion matrices for the different classifiers for various train set sizes
out_file = "results_celebs_extra_sorted_noref_" + data_evaluated[eval_all] + ".txt"

scaler = StandardScaler()
scaler.fit(features)
features_scaled = scaler.transform(features)

features = features_scaled

features_eval = A_test_scaled
y_eval = y_test

for i in range(0,5):
	train_p = train_p_inc + train_p_inc*i
	trn_all, tst_all = split_data(train_p = train_p, seed = SEED)
	train_data = features[trn_all]
	train_label = y[trn_all]
	test_data = features[tst_all]
	test_label = y[tst_all]

	def eval_model(clf, confusion, title, train_data, train_label, features, y):
		
		if (eval_all == 0):
			results = clf.fit(train_data, train_label).predict(train_data)
			cnf = confusion_matrix(train_label, results)
			print("Evaluating models on training data...")
		
		if (eval_all == 1):
			results = clf.fit(train_data, train_label).predict(features)
			cnf = confusion_matrix(y, results)
			#print("\n", title, clf.score(features, y),"\n",cnf)
			print("Evaluating models on training and testing data...")

		confusion.append(cnf)


	# LDA
	clf = LinearDiscriminantAnalysis(n_components=2)
	eval_model(clf, confusion, "---- LDA ----", train_data, train_label, features_eval, y_eval)
	drA = clf.transform(features)
		
	#SVM-LIN
	clf = svm.SVC(kernel='linear', C=0.01)
	eval_model(clf, confusion, "---- SVM-LIN ----", train_data, train_label, features_eval, y_eval)

	#SVM-RBF
	clf = svm.SVC(kernel='rbf', C=0.01)
	eval_model(clf, confusion, "---- SVM-RBF ----", train_data, train_label, features_eval, y_eval)

	#MLP
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	eval_model(clf, confusion, "---- MLP ----", train_data, train_label, features_eval, y_eval)

	#KNN
	clf = neighbors.KNeighborsClassifier(5)
	eval_model(clf, confusion, "---- KNN ----", train_data, train_label, features_eval, y_eval)

np.savetxt(out_file, confusion, fmt="%s")