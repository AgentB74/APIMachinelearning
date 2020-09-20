from rest_framework.serializers import ModelSerializer
from .models import TrainedModel


class TrainedModelSerializer(ModelSerializer):
    class Meta:
        model = TrainedModel
        fields = ('name_model', 'model', 'metrics',)
