import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import tracking.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'human_tracking.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(
        tracking.routing.websocket_urlpatterns
    ),
})
