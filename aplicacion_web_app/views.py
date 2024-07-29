from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from rest_framework.decorators import api_view #Prueba en postman
from . models import *
from datetime import datetime

def login(request):
    return render(request, 'login.html')

# @api_view(['POST'])
def sign_up(request):
    return render(request, 'signup.html')

    # usuario_nuevo = request.data.get('username')
    # correo = request.data.get('email')
    # pass_nuevo = request.data.get('pass')
    # try: 
    #     fecha_sesion_iniciada = datetime.today()
    #     auth_user.objects.create(
    #         password = pass_nuevo,
    #         last_login = fecha_sesion_iniciada,
    #         username = usuario_nuevo,
    #         email = correo
    #     )
    #     return JsonResponse({'Mensaje' : 'Se registro correctamente'}, status = 200)
    # except Exception as e:
    #     print('No se pudo registrar. Error: ', e)
    #     return JsonResponse({'Mensaje' : 'Error interno del servidor'}, status = 500)
 