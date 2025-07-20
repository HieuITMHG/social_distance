from django.urls import path
from . import views

urlpatterns = [
    path('', views.multi_youtube, name='multi_youtube'),
    path('stream/webcam/', views.stream_webcam, name='stream_webcam'),
    path('stream/yt1/', views.stream_yt1, name='stream_yt1'),
    path('stream/yt2/', views.stream_yt2, name='stream_yt2'),
    path('stream/yt3/', views.stream_yt3, name='stream_yt3'),
    path('upload/', views.detect_upload, name='upload_detect'),
    path('detect_image/', views.detect_image, name='detect_image'),
    path('detect_video/', views.detect_video, name='detect_video'),
]