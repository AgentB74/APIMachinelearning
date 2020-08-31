# from django.db import models
from djongo import models


# Create your models here.
class Robot(models.Model):
    robot_data = models.TextField(null=False, default='')
    create_date = models.DateTimeField(auto_now_add=True, auto_now=False)

    def __str__(self):
        return self.id
