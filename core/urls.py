from django.urls import path
from . import views

urlpatterns = [
    path('calculate/', views.calculate.as_view(), name="calculate"),
]