from django.http import JsonResponse


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

# Clear Data Function (drop column, duplicate and None)
def cleaningData(rawData):
    rawData = rawData.drop(['deleteColumn'], axis=1)
    rawData = rawData.drop_duplicates()
    rawData = rawData.dropna()
    cleanedData = rawData.copy()
    return cleanedData

# Pre-processing Cross Validation
def preProcessingData(model, X_train, y_train):
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cvScores = cross_val_score(model, X_train, y_train, cv = kf, scoring='r2')
    return cvScores



# Algoritmo Random Forest (RF)
def randomForest(request):
    if(request.method != 'POST'):
        return JsonResponse({'error': 'Bad request'}, status = 500) 

    file = request.FILES.get('file')
    
    if(file is None):
        return JsonResponse({'error': 'No file provided'}, status = 400)
    
    rawData = pd.read_csv(file, delimiter=',', header=None)

    # Add name in the columns
    nameColumns = ['Timestamp_s', 'Timestamp_us', 'LBA', 'Size', 'Entropy', 'deleteColumn']
    rawData.columns = nameColumns
    print(rawData)
    # Process of cleaning data
    cleanedData = cleaningData(rawData)

    # Separate data in feature (X) and object variable (y), then in train and test
    data = cleanedData
    X = data.drop('Entropy', axis = 1)
    y = data['Entropy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    # Configuration model RF
    model = RandomForestRegressor(n_estimators = 315, max_depth = 10, random_state = 42, criterion = 'squared_error')

    # Pre-processing data with Cross Validation
    cv_scores = preProcessingData(model, X_train, y_train)
    
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred) # This is the accuracy
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)

    # Results
    print("Cross-validation scores (R^2):", cv_scores)
    print("Mean cross-validation R^2:", cv_scores.mean())
    print("Test set R^2:", test_r2)
    print("Test set MSE:", test_mse)
    print("Test set RMSE:", test_rmse)
    print("Test set MAE:", test_mae)
    return JsonResponse({'message': 'File processed successfully'}, status=200)