from django.urls import path
from . import views, machinelearning, RandomForest

urlpatterns = [
    path('',views.loginUser,name='login'),
    path('signUp',views.sign_up,name='signUp'),
    path('index', views.index, name='index'),
    #path('MachineLearning/', machinelearning.MachineLearning, name='MachineLearning'),
    path('randomForest', RandomForest.randomForest, name='randomForest' )
]