from django.http import JsonResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Clear Data Function (drop column, duplicate and None)
def cleaningData(rawData):
    rawData = rawData.drop_duplicates()
    rawData = rawData.dropna()
    cleanedData = rawData.copy()
    return cleanedData

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
    plt.title("Learning Curves for XGBoost Classifier")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Extreme Gradient Boosting (XGBoost) algorithm for classification
def xgBoost(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Bad request'}, status=400)
    
    file = request.FILES.get('file')

    if file is None:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    rawData = pd.read_excel(file, header=None)

    # Add name in the columns
    nameColumns = ['Timestamp_s', 'Timestamp_us', 'LBA', 'Size', 'Entropy', 'Type_ransonware']
    rawData.columns = nameColumns

    # Process of cleaning data
    cleanedData = cleaningData(rawData)
    
    data = cleanedData
    X = data.drop('Type_ransonware', axis=1)

    # Codificar la columna 'Type_ransomware' en valores numéricos
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Type_ransonware'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model parameters
    params = {
        'objective': 'multi:softmax',
        'n_estimators': 4,
        'max_depth': 3,
        'eta': 0.9,
        'num_class': len(data['Type_ransonware'].unique()),
        'eval_metric': 'mlogloss',
        'seed': 42,
        'alpha': 1,
        'lambda': 1,
        'gamma': 1,
        'subsample': 0.2,
        'colsample_bytree': 0.2
    }

    # Using XGBClassifier
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)

    # Making predictions
    y_pred = xgb_model.predict(X_test)

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

    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate cross-accuracy using the scikit-learn compatible model
    cross_val_acc = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy').mean()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Any overlap between train and test data: {np.any(np.in1d(X_train.values, X_test.values))}")

    # Return a response with metrics
    response = {
        'message': 'Model trained!',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Cross-Validated Accuracy': cross_val_acc
    }

    # Call learning curves function
    learningCurves(xgb_model, X_train, y_train, X_test, y_test)

    return JsonResponse(response)