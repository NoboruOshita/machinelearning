from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view #Prueba en postman
from django.contrib.auth.hashers import make_password
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

from django.contrib import messages
from . models import *
from datetime import datetime

def login(request):
    if request.method == 'POST':
        usuario = request.POST.get('username')
        psw = request.POST.get('password') 
        try:
            data =  auth_user.objects.get(username=usuario, password=psw)
            if(data):
                
                JsonResponse({'Mensaje' : 'Usuario encontrado'}, status = 200)
        except auth_user.DoesNotExist:
            JsonResponse({'Mensaje' : 'Usuario no encontrado'}, status = 400)
    return render(request, 'login.html')

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
        
        # GUARDADON EN LA BASE DE DATOS
        try: 
            fecha_sesion_iniciada = datetime.today()
            auth_user.objects.create(
                password = make_password(pass_nuevo),
                last_login = fecha_sesion_iniciada,
                username = usuario_nuevo,
                email = correo
            )
            return JsonResponse({'Mensaje' : 'Se registro correctamente'}, status = 200)
        except Exception as e:
            print('No se pudo registrar. Error: ', e)
            return JsonResponse({'Mensaje' : 'Error interno del servidor'}, status = 500)
    return render(request, 'signup.html')
 