from django.db import models
from django.conf import settings
# Create your models here.

class auth_user(models.Model):
    id=models.BigAutoField(primary_key=True)
    password=models.CharField(max_length=128,null=False,blank=True)
    last_login =models.DateTimeField(null=True, blank=True)
    username =models.CharField(max_length=150, null=False,blank=True)
    email = models.CharField(max_length=150, null=False,blank=True)
    class Meta:
        managed = False
        db_table = 'auth_user'
