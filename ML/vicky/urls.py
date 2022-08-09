from django.urls import path
from . import views

urlpatterns = [
    path('', views.hi),
    path('bye/', views.bye),
    path('api/', views.api),
    path('api2/', views.api2),
    path('api3/', views.api3),
    path('api4/', views.api4)
]
