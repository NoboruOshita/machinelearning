from django.urls import path
from . import views

urlpatterns = [
    # path('',views.home,name='home')
    path('MachineLearning/', views.MachineLearning, name='MachineLearning'),
]