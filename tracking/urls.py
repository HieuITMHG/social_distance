from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('detect_image/', views.detect_image, name='detect_image'),
    path('detect_video/', views.detect_video, name='detect_video'),
    path('upload/', views.upload_detect, name='upload_detect'),
    path('webcam/', views.webcam_detect, name='webcam_detect'),
]