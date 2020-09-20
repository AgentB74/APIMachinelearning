from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from robot.robot_handler import RobotHandler
from .learning import do_learn


# class ModelList(APIView, RobotHandler):
#     """
#     Create a new learning model.
#     """
#
#     def get_object(self, kwargs):
#         try:
#             l_model = LearningModel.objects.get(**kwargs)
#             serializer = LearningModelSerializer(l_model)
#             return serializer
#         except LearningModel.DoesNotExist:
#             return None
#
#     def post(self, request, robot_id, format=None):
#         serializer = self.get_object(request.data)
#         if serializer is None:
#             serializer = LearningModelSerializer(data=request.data)
#             if serializer.is_valid():
#                 serializer.save()
#                 return Response(serializer.data, status=status.HTTP_201_CREATED)
#             else:
#                 return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#         print(serializer.data)
#         # print(self.get_df_data(robot_id))
#         do_learn(serializer.data)
#         return Response(serializer.data, status=status.HTTP_201_CREATED)
