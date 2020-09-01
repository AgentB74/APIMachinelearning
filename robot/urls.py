from django.urls import path
from . import views
from django.conf.urls import url


app_name = 'robot'

urlpatterns = [
    path('add/', views.FileUploadView.as_view(), name='add'),
    url('detail/(?P<pk>[0-9]+)$', views.RobotDetail.as_view()),
]
