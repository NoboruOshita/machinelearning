from django.http import JsonResponse
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
        usuario_nuevo = request.POST.get('username')
        correo = request.POST.get('email')
        pass_nuevo = request.POST.get('password')
        pass_confirmacion = request.POST.get('confpassword')

        #VALIDATION
        if not all([usuario_nuevo, correo, pass_nuevo, pass_confirmacion]):
            context = {'error_message': 'Todos los campos son obligatorios'}
            return render (request, 'appweb/signup.html', context)
        if pass_nuevo != pass_confirmacion:
            context = {'error_message': 'Las contraseñas no coinciden'}
            return render (request, 'appweb/signup.html', context)
        try:
            validate_email(correo)
        except ValidationError:
            context = {'error_message': 'El correo electrónico no es válido'}
            return render (request, 'appweb/signup.html', context)
        if len(pass_nuevo) < 8:
            context = {'error_message': 'La contraseña debe tener al menos 8 caracteres'}
            return render (request, 'appweb/signup.html', context)
        if auth_user.objects.filter(username=usuario_nuevo).exists() or auth_user.objects.filter(email=correo).exists():
            print('HOLA')
            context = {'error_message': 'El nombre de usuario o el correo electrónico ya están en uso'}
            return render (request, 'appweb/signup.html', context)
        
       # SAVE IN DATABASE
        try:
            user = User.objects.create_user(
                username=usuario_nuevo,
                email=correo,
                password=pass_nuevo
            )
            user.save()  # Saves the user in PostgreSQL
            messages.success(request, 'Se registró correctamente')
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
def predicRansomware(request):
    '''
        - Verifica el IP (Black List) almacenada en la BD
        - Si no hay IP en BD, continuar con el funcionamiento de ML
    '''
    rfModel = joblib.load('random_forest_model.pkl')
    xgbModel = joblib.load('xg_boost_model.pkl')
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
            ransomwareType  = blacklist.objects.get(IP=IP).ransomware_type
            predictedFinal = 100
            context = {'ransomware_type': ransomwareType,
                'probability': predictedFinal,
                'date': date
            }
            return render(request,'appweb/detection/detection.html',context)

        # Verificar que todos los campos estén llenos
        if not (timestampS and timestampMS and LBA and size and entropy):
            context = {"error_message": "Tienes que completar todos los campos."}
            return render(request, 'appweb/detection/detection.html', context)

        try:
            # Convertir los campos a los tipos adecuados
            timestampS = int(timestampS)
            timestampMS = int(timestampMS)
            LBA = float(LBA)
            size = float(size)
            entropy = float(entropy)
        except ValueError:
            context = {"error_message": "Por favor, ingresa valores válidos en todos los campos."}
            return render(request, 'appweb/detection/detection.html', context)

        # Verify data with the ML Model
        inputData = np.array([[timestampS, timestampMS, LBA, size, entropy]])
        
        # Prediction Model Random Forest
        predictionRF = rfModel.predict(inputData)
        probabilitiesRF = rfModel.predict_proba(inputData)
        predictedProbabilityRF = np.max(probabilitiesRF) * 100
        print(predictedProbabilityRF)
        
        # Prediction Model Extreme Gradient Boosting
        predictionXGB = xgbModel.predict(inputData)
        probabilitiesXGB = xgbModel.predict_proba(inputData)
        predictedProbabilityXGB = np.max(probabilitiesXGB) * 100
        print(predictedProbabilityXGB)
        
        # Promedio Simple
        predictedFinal = round((predictedProbabilityRF + predictedProbabilityXGB) / 2, 2)

        context = {'ransomware_type': predictionRF[0],
                   'probability': predictedFinal,
                   'date': date
                }

    return render(request,'appweb/detection/detection.html',context)

def responseRansomware(request):
    return