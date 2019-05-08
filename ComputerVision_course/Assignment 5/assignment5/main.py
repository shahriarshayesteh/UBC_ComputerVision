#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch

import numpy as np
from util import sample_images, build_vocabulary, get_bags_of_sifts
from classifiers import nearest_neighbor_classify, svm_classify,plot_confusion_matrix,MLPclassifier,svm_classify2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset.
#You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = sample_images("sift/train", n_sample=600)
test_image_paths, test_labels = sample_images("sift/test", n_sample=200)


''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the
 dimensionality of each image representation. See the starter code for
 each function for more details. '''


print('Extracting SIFT features\n')
#TODO: You code build_vocabulary function in util.py
kmeans = build_vocabulary(train_image_paths, vocab_size=200)

#TODO: You code get_bags_of_sifts function in util.py
train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)

#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.


''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')
#TODO: YOU CODE nearest_neighbor_classify function from classifers.py
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)


print('Using support vector machine to predict test set categories\n')
#TODO: YOU CODE svm_classify function from classifers.py
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)

pred_labels_mlp = MLPclassifier(train_image_feats, train_labels, test_image_feats)

pred_labels_svm2 = svm_classify(train_image_feats, train_labels, test_image_feats)


print('---Evaluation---\n')


knn = accuracy_score(test_labels, pred_labels_knn)
svm = accuracy_score(test_labels, pred_labels_svm)
mlp = accuracy_score(test_labels, pred_labels_mlp)
svm2 = accuracy_score(test_labels, pred_labels_svm2)


print("KNN Classifier accuracy:")
print(knn)
print("SVM Classifiers accuracy:")
print(svm)
print("MLPClassifier Classifiers accuracy:")
print(mlp)
print("SVM(RBF KERNEL) Classifiers accuracy:")
print(svm2)

class_names = ["Bedroom","Coast","Forest","Highway","Industrial","InsideCity","Kitchen","LivingRoom","Mountain","Office"
               ,"OpenCountry","Store","Street","Suburb","TallBuilding",]

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, pred_labels_svm)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization for SVMS')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for SVMS')

plt.show()





# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, pred_labels_knn)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization for KNN')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for KNN')

plt.show()

# Step 3: Build a confusion matrix and score the recognition system for
#         each of the classifiers.
# TODO: In this step you will be doing evaluation.
# 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all.
# 2) Build a Confusion matrix and visualize it.
#   You will need to convert the one-hot format labels back
#   to their category name format.


# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. Can reach .60 with K-NN and
#                     different distance metrics.
#  accuracy ~= .60 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
#                     of clusters, SVM regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense SIFT features.
#  accuracy >= .80 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.


#Helper function for plotting average historam for every scene category on training set

def plot_bar(train_image_feats,train_labels,vocabulary_size):

    hist1 = []
    hist0 = []
    hist2 = []
    hist3 = []
    hist4 = []
    hist5 = []
    hist6 = []
    hist7 = []
    hist8 = []
    hist9 = []
    hist10 = []
    hist11 = []
    hist12= []
    hist13= []
    hist14= []

    for i in range(0,len(train_labels)):

        case = train_labels[i]


        if(case == 1):
            hist1.append(train_image_feats[i])

        if(case == 2):

            hist2.append(train_image_feats[i])

        if(case == 3):

            hist3.append(train_image_feats[i])

        if(case == 4):
            hist4.append(train_image_feats[i])

        if(case == 5):
            hist5.append(train_image_feats[i])

        if(case == 6):
            hist6.append(train_image_feats[i])

        if(case == 7):
            hist7.append(train_image_feats[i])

        if(case == 8):
            hist8.append(train_image_feats[i])

        if(case == 9):
            hist9.append(train_image_feats[i])

        if(case == 10):
            hist10.append(train_image_feats[i])

        if(case == 11):
            hist11.append(train_image_feats[i])

        if(case == 12):
            hist12.append(train_image_feats[i])

        if(case == 13):
            hist13.append(train_image_feats[i])

        if(case == 14):
            hist14.append(train_image_feats[i])
        else:
            hist0.append(train_image_feats[i])


    a=np.average(np.asarray(hist0), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Bedroom ")
    plt.ylabel('average of each vocabulary appears in Bedroom class')
    plt.xlabel('Visual Vocabularies')
    plt.show()

    a=np.average(np.asarray(hist1), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Coast")
    plt.ylabel('average of each vocabulary appears in Coast class')
    plt.xlabel('Visual Vocabularies')
    plt.show()

    a=np.average(np.asarray(hist2), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Forest")
    plt.ylabel('average of each vocabulary appears in Forest class')
    plt.xlabel('Visual Vocabularies')
    plt.show()

    a=np.average(np.asarray(hist3), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Highway")
    plt.ylabel('average of each vocabulary appears in Highway class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist4), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Industrial")
    plt.ylabel('average of each vocabulary appears in Industrial class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist5), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam InsideCidty")
    plt.ylabel('average of each vocabulary appears in InsideCidty class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist6), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Kitchen")
    plt.ylabel('average of each vocabulary appears in Kitchen class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist7), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Living Room ")
    plt.ylabel('average of each vocabulary appears in Living Room class')
    plt.xlabel('Visual Vocabularies')
    plt.show()

    a=np.average(np.asarray(hist8), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Mountain")
    plt.ylabel('average of each vocabulary appears in Mountain class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist9), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Office")
    plt.ylabel('average of each vocabulary appears in Office class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist10), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam OpenCountry")
    plt.ylabel('average of each vocabulary appears in OpenCountry class')
    plt.xlabel('Visual Vocabularies')
    plt.show()

    a=np.average(np.asarray(hist11), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Store")
    plt.ylabel('average of each vocabulary appears in Store class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist12), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Street")
    plt.ylabel('average of each vocabulary appears in Street class')
    plt.xlabel('Visual Vocabularies')
    plt.show()
    a=np.average(np.asarray(hist13), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam Suburb")
    plt.ylabel('average of each vocabulary appears in Suburb class')
    plt.xlabel('Visual Vocabularies')
    plt.show()

    a=np.average(np.asarray(hist14), axis = 0)
    plt.bar( [i for i in range(0,vocabulary_size)],a)
    plt.title("average historam TallBuilding")
    plt.ylabel('average of each vocabulary appears in TallBuilding class')
    plt.xlabel('Visual Vocabularies')
    plt.show()


#plot_bar(train_image_feats,train_labels,250)
