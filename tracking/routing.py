from django.urls import re_path
from tracking import consumers

websocket_urlpatterns = [
    re_path(r'ws/violations/$', consumers.ViolationLogConsumer.as_asgi()),
]