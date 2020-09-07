from djongo import models


# Create your models here.
class LearningModel(models.Model):
    name = models.CharField(max_length=20, null=False, default='name_of_learning_model')
    type = models.CharField(max_length=20, null=False, default='type_of_learning_model')
    model_parameters = models.TextField(null=False, default='')


class TrainedModel(models.Model):
    name_model = models.CharField(max_length=20, null=False, default='name_of_trained_model')
    create_date = models.DateTimeField(auto_now_add=True, auto_now=False)
    model = models.ForeignKey(LearningModel, related_name='trained_model', on_delete=models.CASCADE)
    metrics = models.TextField(null=False, default='')
