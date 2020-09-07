from django.urls import path, re_path
from . import views


app_name = 'robot'

urlpatterns = [
    path('add/', views.FileUploadView.as_view(), name='add'),
    re_path('detail/(?P<pk>[0-9]+)$', views.RobotDetail.as_view(), name='detail'),
]
