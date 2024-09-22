from django.http import JsonResponse #Prueba
from rest_framework.decorators import api_view #Prueba en postman

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
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

def sign_up(request):
    if request.method == 'POST':
        usuario_nuevo = request.POST.get('username')
        correo = request.POST.get('email')
        pass_nuevo = request.POST.get('password')
        pass_confirmacion = request.POST.get('confpassword')

        #VALIDACIÓN
        if not all([usuario_nuevo, correo, pass_nuevo, pass_confirmacion]):
            return JsonResponse({'Mensaje': 'Todos los campos son obligatorios'}, status=400)
        if pass_nuevo != pass_confirmacion:
            return JsonResponse({'Mensaje': 'Las contraseñas no coinciden'}, status=400)
        try:
            validate_email(correo)
        except ValidationError:
            return JsonResponse({'Mensaje': 'El correo electrónico no es válido'}, status=400)
        if len(pass_nuevo) < 8:
            return JsonResponse({'Mensaje': 'La contraseña debe tener al menos 8 caracteres'}, status=400)
        if auth_user.objects.filter(username=usuario_nuevo).exists() or auth_user.objects.filter(email=correo).exists():
            return JsonResponse({'Mensaje': 'El nombre de usuario o el correo electrónico ya están en uso'}, status=400)
        
       # GUARDADO EN LA BASE DE DATOS
        try:
            user = User.objects.create_user(
                username=usuario_nuevo,
                email=correo,
                password=pass_nuevo
            )
            user.save()  # Esto guarda el usuario en PostgreSQL
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
def detection(request):
    return render(request, 'appweb/detection/detection.html')

@login_required
def predicRansomware (request):
    rfModel = joblib.load('random_forest_model.pkl')
    if request.method == 'POST':
        timestampS= int(request.POST.get('Timestamp_s'), 0)
        timestampMS= int(request.POST.get('Timestamp_ms'), 0)
        LBA = float(request.POST.get('LBA', 0))
        size = float(request.POST.get('Size', 0))
        entropy = float(request.POST.get('Entropy', 0))

        inputData = np.array([[timestampS, timestampMS, LBA, size, entropy]])
        prediction = rfModel.predict(inputData)

        probabilities = rfModel.predict_proba(inputData)
        predicted_probability = np.max(probabilities) * 100
        
        context = {'ransomware_type': prediction[0],
                    'probability': predicted_probability
                }

    return render(request, 'appweb/detection/detection.html', context)
