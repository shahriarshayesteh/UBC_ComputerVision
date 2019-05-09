from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import itertools
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

#Starter code prepared by Borna Ghotbi for computer vision
#based on MATLAB code by James Hay

'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.

    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector
        indicating the predicted category for each test image.

    Usefull funtion:

    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''

    # defien a Kneighbors Classifier using sci-kit learn,
    #n_neighbors = number of neighbours that our classifier consider the distance to select the proper class for a test data
    KNNmodel = KNeighborsClassifier(n_neighbors=9)

    # fit the data to our model
    KNNmodel.fit(train_image_feats,train_labels)

    #predict lables of the test data
    predicted_labels = KNNmodel.predict(test_image_feats)


    return predicted_labels



'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.

    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector
        indicating the predicted category for each test image.

    Usefull funtion:

    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html


    '''


    # defien a SVM classifier using sci-kit learn, c= penalty parameter C of the error term
    #Kernel = "rbf",gamma =Kernel coefficient
    svmm = LinearSVC(C =1)
    #given task:1-vs-all linear SVMS to operate in the bag of SIFT feature space
    #so OneVsRestClassifier method 15 binary svms
    clf = OneVsRestClassifier(svmm)
    # fit the data to our model
    clf.fit(train_image_feats,train_labels)
    #predict lables of the test data
    predicted_labels = clf.predict(test_image_feats)

    return predicted_labels


def MLPclassifier(train_image_feats, train_labels, test_image_feats):

    clf = MLPClassifier(hidden_layer_sizes=(256,512,340,150), activation='tanh', learning_rate='adaptive', max_iter=800, alpha=0.008,
                     solver='adam')

    clf.fit(train_image_feats, train_labels)
    predicted_labels = clf.predict(test_image_feats)

    return predicted_labels



def svm_classify2(train_image_feats, train_labels, test_image_feats):

    # defien a SVM classifier using sci-kit learn, c= penalty parameter C of the error term
    #Kernel = "rbf",gamma =Kernel coefficient
    svmm = SVC(C=1, kernel='rbf', degree=6, gamma=0.9)
    #given task:1-vs-all linear SVMS to operate in the bag of SIFT feature space
    #so OneVsRestClassifier method 15 binary svms
    clf = OneVsRestClassifier(svmm)
    # fit the data to our model
    clf.fit(train_image_feats,train_labels)
    #predict lables of the test data
    predicted_labels = clf.predict(test_image_feats)

    return predicted_labels


#Helper function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
