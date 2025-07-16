import cv2
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
from .yolo import detect_and_draw

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.running = True
        self.cap = cv2.VideoCapture(0)
        asyncio.create_task(self.send_frames())

    async def disconnect(self, close_code):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

    async def send_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = detect_and_draw(frame)
            _, jpeg = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            await self.send(text_data=b64)
            await asyncio.sleep(0.03)  # ~30 FPS