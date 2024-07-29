from django.urls import path
from . import views, machinelearning

urlpatterns = [
    path('',views.login,name='login'),
    path('signup/',views.sign_up,name='sign_up')
    #path('MachineLearning/', machinelearning.MachineLearning, name='MachineLearning'),
]