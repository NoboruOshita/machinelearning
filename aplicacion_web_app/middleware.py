from django.utils import translation
from django.utils.deprecation import MiddlewareMixin
from django.contrib.gis.geoip2 import GeoIP2

class GeoIPMiddleware(MiddlewareMixin):
    def process_request(self, request):
        g = GeoIP2()
        ip = request.META.get('REMOTE_ADDR')

        # # Intenta obtener la IP real del cliente
        # ip = request.META.get('HTTP_X_FORWARDED_FOR')
        # if ip:
        #     ip = ip.split(',')[0]  # En caso de múltiples IPs, tomar la primera
        # else:
        #     ip = request.META.get('REMOTE_ADDR')

        # Cambiar a una IP pública para pruebas locales
        if ip == '127.0.0.1':
            ip = '1.0.32.0'  # Cambia esto según tus necesidades
        
        print(f"IP detectada: {ip}")  # Para depuración

        try:
            country_info = g.country(ip)
            country_code = country_info['country_code']
            print(f"País detectado: {country_code}")  # Para depuración
            
            # Seleccionar el código de idioma según el país
            language_code = {
                'US': 'en',
                'GB': 'en',
                'ES': 'es',
                'PE': 'es',
                'DE': 'de',
                'FR': 'fr',
                'JP': 'ja',
                'CN': 'zh-hans',
                'KR': 'ko',
                'PT': 'pt',
                'AE': 'ar',
                'SA': 'ar',
                'EG': 'ar',
            }.get(country_code, 'en')  # Idioma por defecto es 'en'

            translation.activate(language_code)
            request.LANGUAGE_CODE = language_code
            print(f"Idioma activado: {language_code}")  # Para depuración

        except Exception as e:
            print(f"Error al detectar país: {e}")
            translation.activate('en')  # Cambia a un idioma por defecto
            request.LANGUAGE_CODE = 'en'
