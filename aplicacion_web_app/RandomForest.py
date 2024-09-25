from django.http import JsonResponse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Clear Data Function (drop column, duplicate and None)
def cleaningData(rawData):
    rawData = rawData.drop_duplicates()
    rawData = rawData.dropna()
    cleanedData = rawData.copy()
    return cleanedData

# Pre-processing Cross Validation
def preProcessingData(model, X_train, y_train):
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cvScores = cross_val_score(model, X_train, y_train, cv = kf, scoring='accuracy') # Se usa 'accuracy' para clasificación
    return cvScores

# Curvas de Aprendizaje
def learningCurves(model, X_train, y_train, X_test, y_test):
    train_accuracies = []
    test_accuracies = []
    train_sizes = np.linspace(0.1, 0.9, 10)  # Ajustar el rango para evitar 1.0
    
    for train_size in train_sizes:
        X_train_partial, _, y_train_partial, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
        model.fit(X_train_partial, y_train_partial)
        
        # Accuracy on the training set
        y_train_pred = model.predict(X_train_partial)
        train_acc = accuracy_score(y_train_partial, y_train_pred)
        train_accuracies.append(train_acc)
        
        # Accuracy on the test set
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_acc)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_accuracies, label="Training Accuracy", marker='o')
    plt.plot(train_sizes, test_accuracies, label="Test Accuracy", marker='o')
    plt.title("Learning Curves for Random Forest Classifier")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Algoritmo Random Forest (RF)
def randomForest(request):
    if(request.method != 'POST'):
        return JsonResponse({'error': 'Bad request'}, status = 500) 

    file = request.FILES.get('file')
    
    if(file is None):
        return JsonResponse({'error': 'No file provided'}, status = 400)
    
    rawData = pd.read_excel(file, header=None)

    # Add name in the columns
    nameColumns = ['Timestamp_s', 'Timestamp_us', 'LBA', 'Size', 'Entropy', 'Type_ransomware']
    rawData.columns = nameColumns
    print(rawData)
    
    # Process of cleaning data
    cleanedData = cleaningData(rawData)
    
    # Separate data in feature (X) and object variable (y), then in train and test
    data = cleanedData
    X = data.drop('Type_ransomware', axis = 1)
    y = data['Type_ransomware']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    
    # Configuration model RF
    model = RandomForestClassifier(n_estimators = 8, max_depth = 3, random_state = 42, criterion="gini")
    #model = RandomForestClassifier(n_estimators = 4, max_depth = 3, random_state = 42, criterion="entropy")

    # Pre-processing data with Cross Validation
    cv_scores = preProcessingData(model, X_train, y_train)
    
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # Confusion Matrix
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print('Matrix de confusión\n', confusionMatrix)

    TN = confusionMatrix[0][0]
    FP = confusionMatrix[0][1]
    FN = confusionMatrix[1][0]
    TP = confusionMatrix[1][1]
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)
    print("TP: ", TP)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    # Evaluar el modelo en el conjunto de prueba
    test_accuracy = model.score(X_test, y_test)

    # Results
    print("Cross-validation scores (Accuracy):", cv_scores)
    print("Precisión en el conjunto de prueba:", test_accuracy)

    print("Mean cross-validation Accuracy:", cv_scores.mean())
    print("Test set Accuracy:", test_accuracy)
    print("Test set Precision:", test_precision)
    print("Test set Recall:", test_recall)
    print("Test set F1 Score:", test_f1)

    # Plot the learning curves
    learningCurves(model, X_train, y_train, X_test, y_test)
    joblib.dump(model, 'random_forest_model.pkl')
    return JsonResponse({'message': 'File processed successfully'}, status=200)
