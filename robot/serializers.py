from rest_framework.serializers import ModelSerializer
from .models import Robot


class RobotSerializer(ModelSerializer):
    class Meta:
        model = Robot
        fields = ('id', 'robot_data', 'create_date')
