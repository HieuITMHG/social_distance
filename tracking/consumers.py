import json
import time
from channels.generic.websocket import WebsocketConsumer
from tracking.main_V4 import VIOLATION_LOG_QUEUE

class ViolationLogConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        # Gửi log hiện có (nếu cần)
        self.send(text_data=json.dumps({'message': 'Connected to violation log'}))
        
        # Tạo luồng để kiểm tra hàng đợi
        def send_logs():
            while True:
                try:
                    log = VIOLATION_LOG_QUEUE.get_nowait()
                    print("LOG=============================LOG")
                    print(log)
                    print("LOG=============================LOG")
                    self.send(text_data=json.dumps(log))
                except:
                    time.sleep(0.1)  # Ngủ ngắn để tránh busy loop
        
        # Chạy luồng gửi log
        import threading
        threading.Thread(target=send_logs, daemon=True).start()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        # Không cần xử lý dữ liệu từ client trong trường hợp này
        pass