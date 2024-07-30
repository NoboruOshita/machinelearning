from django.urls import path
from . import views, machinelearning

urlpatterns = [
    path('',views.login,name='login'),
    path('sign_up/',views.sign_up,name='sign_up')
    #path('MachineLearning/', machinelearning.MachineLearning, name='MachineLearning'),
]