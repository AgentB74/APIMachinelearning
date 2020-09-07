# from django.conf.urls import url, include
from django.urls import path, re_path
from . import views

app_name = 'learning_model'

urlpatterns = [
    re_path('learn/(?P<robot_id>[0-9]+)$', views.ModelList.as_view(), name='create'),
    # re_path('info/(?P<robot_id>[0-9]+)$', views.Learn.as_view(), name='info'),
]
