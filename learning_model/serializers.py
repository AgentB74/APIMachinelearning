from rest_framework.serializers import ModelSerializer
from .models import LearningModel, TrainedModel


class LearningModelSerializer(ModelSerializer):
    class Meta:
        model = LearningModel
        fields = ('name', 'type', 'model_parameters',)


class TrainedModelSerializer(ModelSerializer):
    class Meta:
        model = TrainedModel
        fields = ('name_model', 'model', 'metrics',)
