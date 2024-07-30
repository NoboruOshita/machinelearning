from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from rest_framework.decorators import api_view #Prueba en postman
from django.contrib import messages
from . models import *
from datetime import datetime

def login(request):
    return render(request, 'login.html')

def sign_up(request):
    if request.method == 'POST':
        usuario_nuevo = request.POST.get('username')
        correo = request.POST.get('email')
        pass_nuevo = request.POST.get('password')
        pass_confirmacion = request.POST.get('confpassword')

        #VALIDACIÓN
    
        # GUARDADON EN LA BASE DE DATOS
        try: 
            fecha_sesion_iniciada = datetime.today()
            auth_user.objects.create(
                password = pass_nuevo,
                last_login = fecha_sesion_iniciada,
                username = usuario_nuevo,
                email = correo
            )
            return JsonResponse({'Mensaje' : 'Se registro correctamente'}, status = 200)
        except Exception as e:
            print('No se pudo registrar. Error: ', e)
            return JsonResponse({'Mensaje' : 'Error interno del servidor'}, status = 500)
    return render(request, 'signup.html')
 