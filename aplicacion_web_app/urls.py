from django.urls import path
from . import views, RandomForest, xgboosting

urlpatterns = [
    path('',views.loginUser,name='login'),
    path('logout_user', views.logout_user,name='logout_user'),
    path('login', views.loginUser, name='login'),
    path('signUp',views.sign_up,name='signUp'),
    path('index', views.index, name='index'),
    path('detections', views.detections, name='detections'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('predicRansomware', views.predicRansomware, name='predicRansomware'),
    path('excelDetail', views.excelDetail, name='excelDetail'),
    path('getExcelflowChart', views.getExcelflowChart, name="getExcelflowChart"),
    path('randomForest', RandomForest.randomForest, name='randomForest' ),
    path('xgBoost', xgboosting.xgBoost, name='xgBoost'),
]