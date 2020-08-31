from django.urls import path
from . import views

app_name = 'robot'

urlpatterns = [
    path('add/', views.FileUploadView.as_view(), name='add'),
]
