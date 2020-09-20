from django.http import QueryDict
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response

import pandas as pd

from learning_model.learning import do_learn
from .robot_handler import RobotHandler
from .serializers import RobotSerializer


class RobotLearnView(APIView, RobotHandler):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        file = request.FILES['file']
        df = pd.read_csv(file)
        df_json = df.to_json()

        file_data = QueryDict('', mutable=True)
        file_data.update({'robot_data': df_json})
        file_serializer = RobotSerializer(data=file_data)

        if file_serializer.is_valid():
            # file_serializer.save()
            # print(df.loc[df['_NumMotor_'] == '_1_'])
            do_learn(robot_data=df.loc[df['_NumMotor_'] == '_1_'])
            return Response('Success', status=status.HTTP_201_CREATED)
        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RobotDetail(APIView, RobotHandler):
    """
    Retrieve, update or delete a snippet instance.
    """

    def get(self, request, pk, format=None):
        robot = self.get_robot(pk)
        serializer = RobotSerializer(robot)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        robot = self.get_robot(pk)
        serializer = RobotSerializer(robot, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        robot = self.get_robot(pk)
        robot.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
