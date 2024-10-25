import json
from django.http import HttpResponse, JsonResponse
from django.utils.translation import gettext as _
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.contrib import messages
from . models import *
from datetime import datetime

import joblib
import numpy as np
import openpyxl
import pandas as pd

from django.db.models import Count
from django.utils.timezone import now, timedelta
# VARIABLES GLOBALS
deviation_entropy = None
deviation_size = None
deviation_LBA = None
predictedProbabilityRF = None
predictedProbabilityXGB = None

resultStorage = {
    'ransomware_type' : [],
    'predictedProbabilityRF': [],
    'predictedProbabilityXGB': [],
    'deviation_entropy': [],
    'deviation_size': [],
    'deviation_LBA': []
}

def loginUser(request):
    '''
        - Login the User
    '''
    if request.method == 'POST':
        usuario = request.POST['username']
        psw = request.POST['password']
        user = authenticate(request, username = usuario, password = psw)
        if user is not None:
            login(request, user)
            messages.success(request, "Inicio de sesión exitoso. ¡Bienvenido!")
            return redirect('index')
        else:
            messages.error(request, "Credenciales inválidas. Por favor, verifica tus credenciales e inténtalo nuevamente.")
            return redirect('login')
    return render(request, 'appweb/login.html')

def logout_user(request):
    logout(request)
    messages.success(request, ("Sesion Cerrada Exitosamente"))
    return redirect('login')


def sign_up(request):
    '''
        - Create new User
        - Save the Data the new User in Database PostgreSQL
    '''
    if request.method == 'POST':
        usuario_nuevo = request.POST.get('username').strip()
        correo = request.POST.get('email').strip()
        pass_nuevo = request.POST.get('password').strip()
        pass_confirmacion = request.POST.get('confpassword').strip()

        #VALIDATION
        if not all([usuario_nuevo, correo, pass_nuevo, pass_confirmacion]):
            context = {'error_message': _('Todos los campos son obligatorios')}
            return render (request, 'appweb/signup.html', context)
        if pass_nuevo != pass_confirmacion:
            context = {'error_message': _('Las contraseñas no coinciden')}
            return render (request, 'appweb/signup.html', context)
        try:
            validate_email(correo)
        except ValidationError:
            context = {'error_message': _('El correo electrónico no es válido')}
            return render (request, 'appweb/signup.html', context)
        if len(pass_nuevo) < 8:
            context = {'error_message': _('La contraseña debe tener al menos 8 caracteres')}
            print(f"Mensaje de error a mostrar: '{context['error_message']}'")
            return render(request, 'appweb/signup.html', context)
        if auth_user.objects.filter(username=usuario_nuevo).exists() or auth_user.objects.filter(email=correo).exists():
            context = {'error_message': _('El nombre de usuario o el correo electrónico ya están en uso')}
            return render (request, 'appweb/signup.html', context)
        
       # SAVE IN DATABASE
        try:
            user = User.objects.create_user(
                username=usuario_nuevo,
                email=correo,
                password=pass_nuevo
            )
            user.save()  # Saves the user in PostgreSQL
            messages.success(request, _('Se registró correctamente'))
            return redirect('login')
        except Exception as e:
            print('No se pudo registrar. Error: ', e)
            return JsonResponse({'Mensaje' : 'Error interno del servidor'}, status = 500)
    return render(request, 'appweb/signup.html')

@login_required
def index(request):
    return render(request, 'appweb/index.html')

@login_required
def detections(request):
    return render(request, 'appweb/detection/detection.html')

@login_required
def dashboard(request):
    end_date = now().date()
    start_date = end_date - timedelta(weeks=4)
    
    # Verificar si el usuario ha especificado una fecha "Desde"
    selected_start = request.GET.get('start_date')
    
    if selected_start:
        start_date = datetime.strptime(selected_start, '%Y-%m-%d').date()

    # Consulta para contar los reales positivos agrupados por día en el rango de fechas
    real_positive_data = (
        response.objects
        .filter(response_date__range=(start_date, end_date))
        .values('response_date')
        .annotate(count=Count('id_response'))
        .order_by('response_date')
    )

    # Formatear los datos para el gráfico
    data = {
        'labels': [item['response_date'].strftime('%d/%m/%Y') for item in real_positive_data],
        'counts': [item['count'] for item in real_positive_data],
    }
    return render(request, 'appweb/dashboard/dashboard.html', {'data': data, 'start_date': start_date})

@login_required
def predicRansomware(request):
    '''
        - Verifica el IP (Black List) almacenada en la BD
        - Si no hay IP en BD, continuar con el funcionamiento de ML
    '''
    rfModel = joblib.load('random_forest_model.pkl')
    xgbModel = joblib.load('xg_boost_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    id_user = request.user.id
    user_instance = auth_user.objects.get(id=id_user)
    id_blacklist_id = None
    print(type(request.user))
    print('user_instance', user_instance)
    if request.method == 'POST':
        timestampS = request.POST.get('Timestamp_s')
        timestampMS = request.POST.get('Timestamp_ms')
        LBA = request.POST.get('LBA')
        size = request.POST.get('Size')
        entropy = request.POST.get('Entropy')
        IP = request.POST.get('IP')
        
        predictedFinal = 0
        date = datetime.today().date().strftime('%m/%d/%Y')

        # Verificar la ip de BD
        if IP != '' and blacklist.objects.filter(IP=IP).exists():
            blacklist_instance = blacklist.objects.get(IP=IP)
            ransomwareType = blacklist_instance.ransomware_type
            id_blacklist = blacklist_instance.id_blacklist
            predictedFinal = 100
            context = {'ransomware_type': ransomwareType,
                'probability': predictedFinal,
                'date': date,
                'motive':  [_("La ip se encuentra dentro del backlist")]
            }

            # Storage in DB
            saveDataRansomware(
                id_user_id=user_instance, 
                id_blacklist_id=blacklist_instance,
                timestamp_s=timestampS, 
                timestamp_ms=timestampMS, 
                lba=LBA, 
                block_size=size, 
                entropy_shannon=entropy
            )            
            return render(request,'appweb/detection/detection.html',context)
             
        # Verificar que todos los campos estén llenos
        if not (timestampS and timestampMS and LBA and size and entropy):
            context = {"error_message": _("Tienes que completar todos los campos.")}
            return render(request, 'appweb/detection/detection.html', context)

        try:
            # Convertir los campos a los tipos adecuados
            timestampS = int(timestampS)
            timestampMS = int(timestampMS)
            LBA = float(LBA)
            size = float(size)
            entropy = float(entropy)
        except ValueError:
            context = {"error_message": _("Por favor, ingresa valores válidos en todos los campos.")}
            return render(request, 'appweb/detection/detection.html', context)

        # Verify data with the ML Model
        inputData = np.array([[timestampS, timestampMS, LBA, size, entropy]])
        
        '''
        Probabilidad estimada por cada modelo (Random Forest y XGBoost):
        Explicación:
            Se calcula la probabilidad final combinada de ambos modelos. Se muestrar estas probabilidades por separado como un valor cuantitativo que respalda la decisión. 
        Ejemplo:
            "Random Forest estima una probabilidad de detección del X%, mientras que XGBoost estima una probabilidad del Y%".
        '''
        global predictedProbabilityRF, predictedProbabilityXGB
        # Prediction Model Random Forest
        probabilitiesRF = rfModel.predict_proba(inputData)
        predictedProbabilityRF = np.max(probabilitiesRF) * 100
        typeRansomwareRF = rfModel.predict(inputData)
        
        # Prediction Model Extreme Gradient Boosting
        probabilitiesXGB = xgbModel.predict_proba(inputData)
        predictedProbabilityXGB = np.max(probabilitiesXGB) * 100
        preTypeRansomwareXGB = xgbModel.predict(inputData)
        typeRansomwareXGB = label_encoder.inverse_transform(preTypeRansomwareXGB)

        '''
        Desviación o anomalía en las características:
        Explicación
            Para cada entrada proporcionada, se calcula la desviación respecto a un valor promedio de las características normales. Así, se justifica el porcentaje en base a cuán lejos están los valores del comportamiento esperado (valores típicos de ransomware vs normales).
        Ejemplo:
            "La entropía observada es un Z% mayor que el promedio de los archivos normales, lo que contribuye a la alta probabilidad de ransomware".
        '''
        # Analyze deviations (anomalies)
        global deviation_entropy, deviation_size, deviation_LBA
        deviation_entropy = round(abs(entropy - 0.95) * 100, 2)  # 0.95 as average entropy threshold
        deviation_size = round(abs(size - 1024) / 100, 2) # 1024 as a threshold for block size
        deviation_LBA = round(abs(LBA - 600000), 2)  # LBA threshold
        
        # Final prediction
        if typeRansomwareRF[0] == typeRansomwareXGB[0]:
            ransomwareType = typeRansomwareXGB[0]
            # Promedio Simple
            predictedFinal = round((predictedProbabilityRF + predictedProbabilityXGB) / 2, 2)
        elif predictedProbabilityRF > predictedProbabilityXGB:
            ransomwareType = typeRansomwareRF[0]
            predictedFinal = round(predictedProbabilityRF, 2)
        else:
            ransomwareType = typeRansomwareXGB[0]
            predictedFinal = round(predictedProbabilityXGB, 2)

        # Evaluate final confidence and response
        if predictedFinal > 70:
            features = {'entropy': entropy, 'size': size, 'LBA': LBA}
            mensaje, motive = responseRansomware(predictedFinal, features)
            # Storage in DB
            data_ransomware_instance =saveDataRansomware(id_user_id=user_instance, id_blacklist_id=None, timestamp_s=timestampS, timestamp_ms=timestampMS, lba=LBA, block_size=size, entropy_shannon=entropy)
        else:
            motive = _('Sin respuesta contra la amenaza')
        
        detection_instance = saveDetection(int(data_ransomware_instance), ransomwareType, predictedFinal, date)
        response_instance = saveResponse (int(detection_instance), motive,date)
        log(int(detection_instance),int(response_instance))
        
        ransomwareType = ransomwareType if predictedFinal > 70 else ""
        global resultStorage
        resultStorage['ransomware_type'].append(ransomwareType)
        resultStorage['predictedProbabilityRF'].append(predictedProbabilityRF)
        resultStorage['predictedProbabilityXGB'].append(predictedProbabilityXGB)
        resultStorage['deviation_entropy'].append(deviation_entropy)
        resultStorage['deviation_size'].append(deviation_size)
        resultStorage['deviation_LBA'].append(deviation_LBA)

        context = {
            'ransomware_type': ransomwareType,
            'probability': predictedFinal,
            'date': date,
            'motive': motive,
            'predictedProbabilityRF': round(predictedProbabilityRF,2),
            'predictedProbabilityXGB': round(predictedProbabilityXGB,2),
            'deviation_entropy': deviation_entropy,
            'deviation_size': deviation_size,
            'deviation_LBA': deviation_LBA
        }

    return render(request, 'appweb/detection/detection.html', context)

def responseRansomware(predictedFinal, features):
    '''
        - Analisis de umbrales de cada caracteristicas (Anomalia)
    '''
    motivos = []
    if features['entropy'] > 0.95:
        motivos.append(_("Se detectó anomalía en la entropía de Shannon"))
    
    if features['size'] > 1024:
        motivos.append(_("El tamaño del bloque es inusualmente grande"))
    
    if features['LBA'] > 500000:
        motivos.append(_("Acceso a bloques sospechosos de memoria"))
    
    if not motivos:
        motivos.append(_("No se detectaron anomalías significativas en las características observadas"))

    mensaje = f"El documento fue puesto en cuarentena. Motivos:\n"
    for motivo in motivos:
        mensaje += f"- {motivo}\n"
    
    return mensaje, motivos

# Storages in DataBase
def saveDataRansomware(id_user_id, id_blacklist_id, timestamp_s, timestamp_ms, lba, block_size, entropy_shannon):
    data = data_ransomware(
        id_user = id_user_id,
        timestamp_s= timestamp_s,
        timestamp_ms= timestamp_ms,
        lba = lba,
        block_size = block_size,
        entropy_shannon = entropy_shannon
    )

    if id_blacklist_id:
        data = data_ransomware(
            id_user = id_user_id,
            id_blacklist = id_blacklist_id,
            timestamp_s= None,
            timestamp_ms= None,
            lba = None,
            block_size = None,
            entropy_shannon = None
        )
    
    data.save()

    return data.id_data_ransomware

def saveDetection(data_ransomware_instance, ransomwareType, predictedFinal, date):
    ransomware_detected = data_ransomware_instance is not None
    if predictedFinal < 50: ransomwareType = None
    date = datetime.strptime(date, '%m/%d/%Y').date()
    data = detection(
        id_data_ransomware=data_ransomware.objects.get(id_data_ransomware=data_ransomware_instance),
        ransomware_detected=ransomware_detected,
        ransomware_type=ransomwareType,
        percentage_reliability=predictedFinal,
        detection_date=date
    )
    data.save()

    return data.id_detection

def saveResponse(detection_instance, motive, date):
    date = datetime.strptime(date, '%m/%d/%Y').date()
    action = "cuarentena" if motive else None
    detail = ','.join(motive).replace(',', '. ')
    data = response(
        id_detection=detection.objects.get(id_detection=detection_instance),
        action=action,
        detail=detail,
        response_date= date
    )
    data.save()
    
    return data.id_response

def log(detection_instance, response_instance):
    log = logs(
        id_detection=detection.objects.get(id_detection=detection_instance),
        id_response=response.objects.get(id_response=response_instance)
    )
    log.save()

@login_required
def excelDetail(request):
    date = datetime.today()
    formatDate = datetime.strftime(date, '%d/%m/%y')
    formatHours = datetime.now().strftime("%H:%M:%S")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    entityName = {
        'Deviation Entropy': _("Desviación de Entropía"),
        'Deviation size': _("Desviación de Tamaño"),
        'Deviation LBA': _("Desviacion de Dirección de bloque lógico")
    }
    global resultStorage, deviation_size, deviation_LBA, deviation_entropy

    # Column headings
    headers = [
        _("Fecha"),
        _("Hora"),
        _("Ransomware"),
        _("Probabilidad de predicción de Random Forest"), 
        _("Probalidad de predicción de XGBoost"), 
        entityName['Deviation size'], 
        entityName['Deviation LBA'], 
        entityName['Deviation Entropy'],
    ]

    # Add headers to Excel 
    sheet.append(headers)
    
    # Add the date to each row of data 
    rows = [] 
    for i in range(len(resultStorage['predictedProbabilityRF'])): 
        # Add the data for each “input” to a row 
        row = [ formatDate,
               formatHours,
               resultStorage['ransomware_type'][i],
               resultStorage['predictedProbabilityRF'][i], 
               resultStorage['predictedProbabilityXGB'][i],
               resultStorage['deviation_size'][i], 
               resultStorage['deviation_LBA'][i], 
               resultStorage['deviation_entropy'][i] ] 
        # Add the row to the row list 
        rows. append(row) 
    
    # Add all rows to Excel.
    for row in rows:
        sheet.append(row)
    

    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    filename = _("Detalle del resultado") + ".xlsx"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    workbook.save(response)

    print("¡Archivo Excel creado exitosamente!")

    #Clear resultStorage
    for key in resultStorage:
        resultStorage[key].clear()
        
    return response
