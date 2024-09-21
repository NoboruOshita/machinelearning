from django.urls import path
from . import views, machinelearning, RandomForest, xgboosting

urlpatterns = [
    path('',views.loginUser,name='login'),
    path('signUp',views.sign_up,name='signUp'),
    path('index', views.index, name='index'),
    path('detection', views.detection, name='detection'),
    #path('MachineLearning/', machinelearning.MachineLearning, name='MachineLearning'),
    path('randomForest', RandomForest.randomForest, name='randomForest' ),
    path('xgBoost', xgboosting.xgBoost, name='xgBoost')
]