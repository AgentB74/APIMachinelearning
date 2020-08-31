from django.http import QueryDict
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response

import pandas as pd
from .serializers import RobotSerializer


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        file = request.data['file']
        df = pd.read_csv(file)
        df_json = df.to_json()

        file_data = QueryDict('', mutable=True)
        file_data.update({'robot_data': df_json})
        file_serializer = RobotSerializer(data=file_data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response('Success', status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
