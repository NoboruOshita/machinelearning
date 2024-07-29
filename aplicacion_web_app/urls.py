from django.urls import path
from . import views, machinelearning

urlpatterns = [
    #path('',views.login,name='login')
    path('MachineLearning/', machinelearning.MachineLearning, name='MachineLearning'),
]