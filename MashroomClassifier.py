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

encoder = dict();
for i in range(ord('a'), ord('z') + 1):
    encoder[chr(i)] = i - 96;
encoder['?'] = 27;

def prepareTrainAndTestData(path):

    fileIn = open(path, "r", encoding='utf-8');
    lines = fileIn.read().split('\n');

    trainFeatureValues = [];
    trainClasses = [];
    testFeatureValues = [];
    testClasses = [];
    trainStart = 1;
    trainEnd = int((len(lines)*0.7));
    testStart = trainEnd;
    testEnd = len(lines);
    # for i in range(1,27):
    #     print("dict " + str(encoder[chr(i+96)]));


    for i in range(trainStart,trainEnd):
        data = lines[i].split(',');
        trainClasses.append(data[0]);
        lengthData = len(data);
        values = [];
        # print(encoder.values());
        for j in range(1,lengthData):
            # print("data " + data[j]);
            values.append(encoder[data[j]]);
        trainFeatureValues.append(values);

    for i in range(testStart, testEnd):
        data = lines[i].split(',');
        testClasses.append(data[0]);
        lengthData = len(data);
        values = [];
        for j in range(1, lengthData):
            values.append(encoder[data[j]]);
        testFeatureValues.append(values);
    return trainFeatureValues,trainClasses,testFeatureValues,testClasses;

def encodedArray(array):
    array2 = []
    for i in range(0,len(array)):
        # print(array[i])
        array2.append(encoder[array[i]])
        # print(array[i])
    return array2

def decisionTree(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses=prepareTrainAndTestData(path);
    clf = tree.DecisionTreeClassifier(class_weight="balanced",max_depth=15);
    print(clf)
    clf.fit(trainFeatureValues,trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("-----------------------------------------------------------------")
    print("Decision tree accuracy score = " + str(accuracy_score(testClasses, predictedClasses) * 100) + "%");
    print("Decision tree precision score = " + str(
        precision_score(encodedArray(testClasses), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("Decision tree recall score = " + str(
        recall_score(encodedArray(testClasses), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("Decision tree f1 score = " + str(
        f1_score(encodedArray(testClasses), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")
    dot_data = tree.export_graphviz(clf, out_file='tree.txt')
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # graph.write_pdf("iris.pdf")

def svmC(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);

    zipped = list(zip(testClasses, testFeatureValues))
    random.shuffle(zipped)
    testClasses, testFeatureValues = zip(*zipped)

    testClasses2 = testClasses[:int(len(testClasses) * .5)]
    validationClasses = testClasses[int(len(testClasses) * .5) + 1:]
    testFeatureValues2 = testFeatureValues[:int(len(testClasses) * .5)]
    validationFeatureValues = testFeatureValues[int(len(testClasses) * .5) + 1:]

    clf = svm.SVC(kernel="poly",cache_size=1000);
    print(clf)
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(validationFeatureValues);
    print("-----------------------------------------------------------------")
    print("--------------------VALIDATION--------------------------")
    print("SVM accuracy score = " + str(accuracy_score(validationClasses, predictedClasses) * 100) + "%");
    print("SVM precision score = " + str(
        precision_score(encodedArray(validationClasses), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("SVM recall score = " + str(
        recall_score(encodedArray(validationClasses), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("SVM f1 score = " + str(
        f1_score(encodedArray(validationClasses), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")
    predictedClasses = clf.predict(testFeatureValues2);
    print("-----------------------------------------------------------------")
    print("--------------------TESTING--------------------------")
    print("SVM accuracy score = " + str(accuracy_score(testClasses2, predictedClasses) * 100) + "%");
    print("SVM precision score = " + str(
        precision_score(encodedArray(testClasses2), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("SVM recall score = " + str(
        recall_score(encodedArray(testClasses2), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("SVM f1 score = " + str(
        f1_score(encodedArray(testClasses2), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")


def naiveByes(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);
    clf = GaussianNB();
    print(clf);
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("-----------------------------------------------------------------")
    print("Naive Byes accuracy score = " + str(accuracy_score(testClasses, predictedClasses) * 100) + "%");
    print("Naive Byes precision score = " + str(precision_score(encodedArray(testClasses), encodedArray(predictedClasses),average='macro') * 100) + "%");
    print("Naive Byes recall score = " + str(recall_score(encodedArray(testClasses), encodedArray(predictedClasses),average='macro') * 100) + "%");
    print("Naive Byes f1 score = " + str(f1_score(encodedArray(testClasses), encodedArray(predictedClasses),average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")


def neuralNetwork(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);

    zipped = list(zip(testClasses, testFeatureValues))
    random.shuffle(zipped)
    testClasses, testFeatureValues = zip(*zipped)

    testClasses2 = testClasses[:int(len(testClasses)*.5)]
    validationClasses = testClasses[int(len(testClasses)*.5)+1:]
    testFeatureValues2 = testFeatureValues[:int(len(testClasses) * .5)]
    validationFeatureValues = testFeatureValues[int(len(testClasses) * .5) + 1:]
    clf = MLPClassifier(activation='relu', alpha=.0001,
       epsilon=1e-08, hidden_layer_sizes=(100,100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver="adam",tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    print(clf);
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(validationFeatureValues);
    print("-----------------------------------------------------------------")
    print("--------------------VALIDATION--------------------------")
    print("Neural network accuracy score = " + str(accuracy_score(validationClasses, predictedClasses) * 100) + "%");
    print("Neural network precision score = " + str(precision_score(encodedArray(validationClasses), encodedArray(predictedClasses),average='macro') * 100) + "%");
    print("Neural network recall score = " + str(recall_score(encodedArray(validationClasses), encodedArray(predictedClasses),average='macro') * 100) + "%");
    print("Neural network f1 score = " + str(f1_score(encodedArray(validationClasses), encodedArray(predictedClasses),average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")
    predictedClasses = clf.predict(testFeatureValues2);
    print("-----------------------------------------------------------------")
    print("--------------------TESTING--------------------------")
    print("Neural network accuracy score = " + str(accuracy_score(testClasses2, predictedClasses) * 100) + "%");
    print("Neural network precision score = " + str(
        precision_score(encodedArray(testClasses2), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("Neural network recall score = " + str(
        recall_score(encodedArray(testClasses2), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("Neural network f1 score = " + str(
        f1_score(encodedArray(testClasses2), encodedArray(predictedClasses), average='macro') * 100) + "%");
    print("-----------------------------------------------------------------")

path = "mushrooms.csv";
naiveByes(path);
svmC(path);
decisionTree(path);
neuralNetwork(path)


