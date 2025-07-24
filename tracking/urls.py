from django.urls import path
from . import views

urlpatterns = [
    path('', views.multi_youtube, name='multi_youtube'),
    path('stream/cam1/', views.stream_cam1, name='stream_cam1'),
    path('stream/cam2/', views.stream_cam2, name='stream_cam2'),
    path('upload/', views.detect_upload, name='upload_detect'),
    path('detect_image/', views.detect_image, name='detect_image'),
    path('detect_video/', views.detect_video, name='detect_video'),
    path('stream/logs/', views.stream_violation_logs, name='stream_violation_logs'),
]