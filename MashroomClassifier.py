from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pydotplus
from IPython.display import Image
from sklearn.neural_network import MLPClassifier
import random
from sklearn import preprocessing
import numpy as np
encoder = dict();
for i in range(ord('a'), ord('z') + 1):
    encoder[chr(i)] = i - 96;
encoder['?'] = 27;

def prepareTrainAndTestData(path):

    fileIn = open(path, "r", encoding='utf-8');
    lines = fileIn.read().split('\n');

    allEValues = []
    allEClasses = []
    allPValues = []
    allPClasses = []
    # for i in range(1,27):
    #     print("dict " + str(encoder[chr(i+96)]));
    for i in range(1,len(lines)):
        data = lines[i].split(',')
        if(data[0] == 'e'):
            allEClasses.append(data[0])
            lengthData = len(data)
            values = []
            for j in range(1, lengthData):
                values.append(encoder[data[j]])
            allEValues.append(values)
        else:
            allPClasses.append(data[0])
            lengthData = len(data)
            values = []
            for j in range(1, lengthData):
                values.append(encoder[data[j]])
            allPValues.append(values)


    elen = len(allEValues);

    # features = allEValues;
    # features+= allPValues;
    # features_scaled = preprocessing.scale(np.array(features));
    # allEValues = list(features_scaled[:elen-1]);
    # allPValues = list(features_scaled[elen:]);
    #print(allEValues);



    #print(len(allPValues),len(allEValues))
    #print(allEValues);
    trainClasses = allEClasses[:int(len(allEClasses)*0.7)]
    trainFeatureValues = allEValues[:int(len(allEValues)*0.7)]
    #
    trainClasses += allPClasses[:int(len(allPClasses)*0.7)]
    trainFeatureValues += allPValues[:int(len(allPValues) * 0.7)]
    #print(trainFeatureValues);
    testClasses = allEClasses[int(len(allEClasses) * 0.7)+1:]
    testFeatureValues = allEValues[int(len(allEValues) * 0.7)+1:]
    testClasses += allPClasses[int(len(allPClasses) * 0.7)+1:]
    testFeatureValues += allPValues[int(len(allPValues) * 0.7)+1:]

    zipped = list(zip(trainClasses, trainFeatureValues))
    random.shuffle(zipped)
    trainClasses, trainFeatureValues = zip(*zipped)
    trainFeatureValues = list(trainFeatureValues);
    trainClasses = list(trainClasses);
    #print(trainFeatureValues);
    zipped = list(zip(testClasses, testFeatureValues))
    random.shuffle(zipped)
    testClasses, testFeatureValues = zip(*zipped)
    testFeatureValues = list(testFeatureValues);
    testClasses = list(testClasses);

    return trainFeatureValues,trainClasses,testFeatureValues,testClasses;

def decisionTree(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses=prepareTrainAndTestData(path);
    clf = tree.DecisionTreeClassifier(class_weight="balanced",max_depth=20,max_features = "log2",min_samples_split=5,random_state =251254);
    print(clf)
    clf.fit(trainFeatureValues,trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("-----------------------------------------------------------------")
    print("Decision tree accuracy score = " + str(accuracy_score(testClasses, predictedClasses) * 100) + "%");
    print("Decision tree precision score = " + str(
        precision_score(testClasses, predictedClasses, average='macro') * 100) + "%");
    print("Decision tree recall score = " + str(
        recall_score(testClasses, predictedClasses, average='macro') * 100) + "%");
    print("Decision tree f1 score = " + str(
        f1_score(testClasses, predictedClasses, average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")
    dot_data = tree.export_graphviz(clf, out_file='tree.txt')
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # graph.write_pdf("iris.pdf")

def svmC(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);
    #print(trainFeatureValues);
    testClasses2 = testClasses[:int(len(testClasses) * .5)]
    validationClasses = testClasses[int(len(testClasses) * .5) + 1:]
    testFeatureValues2 = testFeatureValues[:int(len(testClasses) * .5)]
    validationFeatureValues = testFeatureValues[int(len(testClasses) * .5) + 1:]

    from sklearn.preprocessing import StandardScaler;
    scaler = StandardScaler()
    scaler.fit(np.array(trainFeatureValues))
    trainFeatureValues_scale = scaler.transform(trainFeatureValues)
    testFeatureValues2_scale = scaler.transform(testFeatureValues2);
    validationFeatureValues_scale = scaler.transform(validationFeatureValues);
    clf = svm.SVC(kernel="linear",C=25);
    print(clf)
    #print(trainFeatureValues);
    clf.fit(trainFeatureValues_scale, trainClasses);
    predictedClasses = clf.predict(validationFeatureValues_scale);
    print("-----------------------------------------------------------------")
    print("--------------------VALIDATION--------------------------")
    print("SVM accuracy score = " + str(accuracy_score(validationClasses, predictedClasses) * 100) + "%")
    print("SVM precision score = " + str(
        precision_score(validationClasses, predictedClasses,average='macro') * 100) + "%")
    print("SVM recall score = " + str(
        recall_score(validationClasses, predictedClasses, average='macro')* 100) + "%")
    print("SVM f1 score = " + str(
        f1_score(validationClasses, predictedClasses, average='macro') * 100) + "%")
    print("-----------------------------------------------------------------")
    predictedClasses = clf.predict(testFeatureValues2_scale);
    print("-----------------------------------------------------------------")
    print("--------------------TESTING--------------------------")
    print("SVM accuracy score = " + str(accuracy_score(testClasses2, predictedClasses) * 100) + "%");
    print("SVM precision score = " + str(
        precision_score(testClasses2, predictedClasses, average='macro') * 100) + "%");
    print("SVM recall score = " + str(
        recall_score(testClasses2, predictedClasses, average='macro') * 100) + "%");
    print("SVM f1 score = " + str(
        f1_score(testClasses2, predictedClasses, average='macro') * 100) + "%");
    print("-----------------------------------------------------------------");


def naiveByes(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);
    clf = GaussianNB();
    print(clf);
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("-----------------------------------------------------------------")
    print("Naive Byes accuracy score = " + str(accuracy_score(testClasses, predictedClasses) * 100) + "%");
    print("Naive Byes precision score = " + str(precision_score(testClasses,predictedClasses,average='macro') * 100) + "%");
    print("Naive Byes recall score = " + str(recall_score(testClasses,predictedClasses,average='macro') * 100) + "%");
    print("Naive Byes f1 score = " + str(f1_score(testClasses,predictedClasses,average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")


def neuralNetwork(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);

    testClasses2 = testClasses[:int(len(testClasses)*.5)]
    validationClasses = testClasses[int(len(testClasses)*.5)+1:]
    testFeatureValues2 = testFeatureValues[:int(len(testClasses) * .5)]
    validationFeatureValues = testFeatureValues[int(len(testClasses) * .5) + 1:]
    clf = MLPClassifier(activation='logistic', alpha=.0001, hidden_layer_sizes=(20,20,20,),random_state=False)
    print(clf);
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(validationFeatureValues);
    print("-----------------------------------------------------------------")
    print("--------------------VALIDATION--------------------------")
    print("Neural network accuracy score = " + str(accuracy_score(validationClasses, predictedClasses) * 100) + "%");
    print("Neural network precision score = " + str(precision_score(validationClasses, predictedClasses,average='macro') * 100) + "%");
    print("Neural network recall score = " + str(recall_score(validationClasses, predictedClasses,average='macro') * 100) + "%");
    print("Neural network f1 score = " + str(f1_score(validationClasses, predictedClasses,average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")
    predictedClasses = clf.predict(testFeatureValues2);
    print("-----------------------------------------------------------------")
    print("--------------------TESTING--------------------------")
    print("Neural network accuracy score = " + str(accuracy_score(testClasses2, predictedClasses) * 100) + "%");
    print("Neural network precision score = " + str(
        precision_score(testClasses2, predictedClasses, average='macro') * 100) + "%");
    print("Neural network recall score = " + str(
        recall_score(testClasses2, predictedClasses, average='macro') * 100) + "%");
    print("Neural network f1 score = " + str(
        f1_score(testClasses2, predictedClasses, average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")

path = "mushrooms.csv";
naiveByes(path);
svmC(path);
decisionTree(path);
neuralNetwork(path)


