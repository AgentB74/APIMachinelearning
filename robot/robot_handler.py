from django.http import Http404

from .models import Robot
import pandas as pd


class RobotHandler:
    @staticmethod
    def get_robot(pk):
        try:
            return Robot.objects.get(pk=pk)
        except Robot.DoesNotExist:
            raise Http404

    def get_json_data(self, pk):
        return self.get_robot(pk).robot_data

    def get_df_data(self, pk):
        return pd.read_json(self.get_robot(pk).robot_data)
