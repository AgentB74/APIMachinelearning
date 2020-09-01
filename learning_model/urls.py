from django.conf.urls import url, include
from django.urls import path
from . import views

app_name = 'learning_model'

urlpatterns = [
    url('info/(?P<robot_id>[0-9]+)$', views.Learn.as_view(), name='info'),
]
