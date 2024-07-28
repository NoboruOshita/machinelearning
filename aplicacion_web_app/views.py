from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from rest_framework.decorators import api_view

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from river import stream, metrics


#Limpieza de datos de datos
def limpiezaDatos(dato_original):
    dato_original = dato_original.drop(['Timestamp_s', 'Target'], axis=1) # Eliminar la columna Target
    dato_original = dato_original.drop_duplicates()
    dato_original = dato_original.dropna()
    data = dato_original.copy()
    return data

#Preprocesamientos de datos
def preprocesamiento_datos(data):
    columnas = ['Timestamp_us', 'LBA', 'Size', 'Entropy']
    mms = MinMaxScaler()
    mms.fit(data[columnas])
    data_mms = mms.transform(data[columnas])

    data_mms = pd.DataFrame(data=data_mms, columns=columnas)
    print(data_mms)

    return data_mms

# Entrenamiento de modelos
def modeloRF(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=9)
    rf_model.fit(X_train, y_train)
    return rf_model

def modeloGB(X_train, y_train):
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model
    
# Load the CSV file with correct delimiter
@api_view(['POST'])
def MachineLearning(request):
    try:
        new_file = request.data.get('file')
        dato_original = pd.read_csv(new_file, delimiter=';')
        
        # Assuming the columns are Timestamp_s, Timestamp_us, LBA, Size, Entropy, Target
        dato_original.columns = ['Timestamp_s', 'Timestamp_us', 'LBA', 'Size', 'Entropy', 'Target']
        
        # Preprocess the data (e.g., normalization, handling missing values if needed)
        # For simplicity, assuming data is clean
        data = limpiezaDatos (dato_original)
        print(data.info())
        print('Cantidad de datos duplicados:\n',data.duplicated().sum())
        print('Cantidad de datos con null:\n', data.isnull().sum(axis=1))
        print('Los datos del archivo\n', data)
        
        datos_modelo = preprocesamiento_datos(data).copy()

        # Define features and target
        X = datos_modelo.drop(columns=['Entropy'], axis=1)
        y = datos_modelo['Entropy']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf_model = modeloRF(X_train, y_train)
        gb_model = modeloGB(X_train, y_train)  


        # Predict and evaluate RandomForest model
        y_pred_rf = rf_model.predict(X_test)
        matrix_confusion = confusion_matrix(y_test, y_pred_rf)

        print("Matrix Confusion en Random Forest\n", matrix_confusion)

        print('Prediction RF:\n', y_pred_rf)
        print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
        print("RandomForest Classification Report:")
        print(classification_report(y_test, y_pred_rf))
        
        breakpoint()
        # Predict and evaluate GradientBoosting model
        y_pred_gb = gb_model.predict(X_test)
        print("GradientBoosting Accuracy:", accuracy_score(y_test, y_pred_gb))
        print("GradientBoosting Classification Report:")
        print(classification_report(y_test, y_pred_gb))
        
        # Simulate response (blocking process) - example
        def simulate_response(predictions):
            actions = []
            for pred in predictions:
                if pred == 1:  # Assuming 1 indicates ransomware
                    actions.append("Block process")
                else:
                    actions.append("Allow process")
            return actions
        
        # Simulate response for RandomForest predictions
        response_rf = simulate_response(y_pred_rf)
        print("RandomForest Response:", response_rf)
        
        # Simulate response for GradientBoosting predictions
        response_gb = simulate_response(y_pred_gb)
        print("GradientBoosting Response:", response_gb)

        return JsonResponse({'Mensaje' : 'Modelos entrenado correctamente'}, status = 200)
    except Exception as e:
        return JsonResponse({'Mensaje' : f'Error. {e}'}, status = 500)