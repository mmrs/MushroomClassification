from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

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

    encoder =  dict();

    for i in range(ord('a'),ord('z')+1):
        encoder[chr(i)]= i-96;
    encoder['?'] = 27;
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



def decissionTree(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses=prepareTrainAndTestData(path);
    clf = tree.DecisionTreeClassifier(class_weight="balanced",max_depth=15);
    print(clf)
    clf.fit(trainFeatureValues,trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("Decission Tree Accuracy Score = " + str(accuracy_score(testClasses,predictedClasses)*100) + "%.");

def svmC(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);
    clf = svm.SVC(kernel="linear",cache_size=1000,class_weight='balanced');
    print(clf)
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("SVM Accuracy Score = " + str(accuracy_score(testClasses, predictedClasses) * 100) + "%.");


def naiveByes(path):
    trainFeatureValues, trainClasses, testFeatureValues, testClasses = prepareTrainAndTestData(path);
    clf = clf = GaussianNB();
    # print(clf);
    clf.fit(trainFeatureValues, trainClasses);
    predictedClasses = clf.predict(testFeatureValues);
    print("Naive Byes Accuracy Score = " + str(accuracy_score(testClasses, predictedClasses) * 100) + "%.");

path = "mushrooms.csv";
decissionTree(path);
# svmC(path);
# naiveByes(path);


