from django.http import JsonResponse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Clear Data Function (drop column, duplicate and None)
def cleaningData(rawData):
    rawData = rawData.drop_duplicates()
    rawData = rawData.dropna()
    # Convertir datos continuos en etiquetas binarias
    rawData['Target'] = (rawData['Entropy'] > 0.3).astype(int)  # Ajusta el umbral según tu necesidad
    # Eliminar la última columna que no es necesaria
    rawData = rawData.drop('deleteColumn', axis=1)
    cleanedData = rawData.copy()
    return cleanedData

# Algoritmo de Extreme Gradient Boosting (XGBoost) para clasificación
def xgBoost(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Bad request'}, status=400)
    
    file = request.FILES.get('file')

    if file is None:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    rawData = pd.read_csv(file, delimiter=',', header=None)

    # Add name in the columns
    nameColumns = ['Timestamp_s', 'Timestamp_us', 'LBA', 'Size', 'Entropy', 'deleteColumn']
    rawData.columns = nameColumns

    # Process of cleaning data
    cleanedData = cleaningData(rawData)
    
    data = cleanedData
    X = data.drop('Target', axis=1)
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Parámetros del modelo con ajustes para reducir sobreajuste
    params = {
        'objective': 'binary:logistic',  # Para clasificación binaria
        'max_depth': 2,                  # Reduce la profundidad del árbol
        'eta': 0.10,                     # Reduce la tasa de aprendizaje
        'eval_metric': 'logloss',
        'seed': 42,
        'alpha': 10,                      # Regularización L1
        'lambda': 10,                     # Regularización L2
        'gamma': 1,                      # Reducción mínima en la pérdida para hacer particiones
        'subsample': 0.8,                # Fracción de datos a usar en cada árbol
        'colsample_bytree': 0.8          # Fracción de columnas a usar en cada árbol
    }

    # Número de rondas de entrenamiento
    numRound = 5  # Ajustar si es necesario

    # Entrenamiento del modelo con early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, numRound, evals, early_stopping_rounds=5)

    # Hacer predicciones
    y_pred_prob = bst.predict(dtest)
    y_pred = [1 if prob > 0.3 else 0 for prob in y_pred_prob]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Evaluar precisión cruzada usando un modelo compatible
    cross_val_model = xgb.XGBClassifier(**params)
    cross_val_acc = cross_val_score(cross_val_model, X, y, cv=5, scoring='accuracy').mean()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Verifica que no haya superposición
    print(f"Any overlap between train and test data: {np.any(np.in1d(X_train.values, X_test.values))}")

    # Devolver una respuesta con las métricas
    response = {
        'message': 'Model trained!',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Cross-Validated Accuracy': cross_val_acc
    }
    return JsonResponse(response)
