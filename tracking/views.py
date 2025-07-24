import threading
import time
import cv2
import numpy as np
import base64
import tempfile
import os
from PIL import Image
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from tracking import config_system
from tracking import main_V4
import logging
import traceback
import torch
from contextlib import contextmanager
import json
from tracking.main_V4 import VIOLATION_LOG_QUEUE

logger = logging.getLogger("detect_upload")

def stream_violation_logs(request):
    def log_generator():
        # Initial message
        yield 'data: {"message": "Connected to violation log"}\n\n'
        while True:
            try:
                # Get log from queue without blocking
                log = VIOLATION_LOG_QUEUE.get_nowait()
                # Format log as SSE data
                log_data = json.dumps(log)
                print(log_data)
                yield f'data: {log_data}\n\n'
            except:
                # Sleep briefly to avoid busy loop
                time.sleep(0.1)
                yield ': keepalive\n\n'  # Send keepalive to maintain connection

    response = StreamingHttpResponse(
        log_generator(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable buffering in proxies like Nginx
    return response

CAMERA_CONFIGS = [
    main_V4.CameraConfig(
        camera_id='cam1', source='0', position='Camera 1',
        enable_recording=config_system.ENABLE_RECORDING,
        recording_path=config_system.RECORDING_PATH,
        confidence_threshold=config_system.CONFIDENCE_THRESHOLD,
        social_distance_threshold=config_system.SOCIAL_DISTANCE_THRESHOLD,
        warning_duration=config_system.WARNING_DURATION,
        loop_video=False
    ),
    main_V4.CameraConfig(
        camera_id='cam2', source=config_system.CAMERA_ID_2, position='Camera 2',
        enable_recording=config_system.ENABLE_RECORDING,
        recording_path=config_system.RECORDING_PATH,
        confidence_threshold=config_system.CONFIDENCE_THRESHOLD,
        social_distance_threshold=config_system.SOCIAL_DISTANCE_THRESHOLD,
        warning_duration=config_system.WARNING_DURATION,
        loop_video=True
    ),
]

# --- Singleton với model YOLOv5 chung ---
class MultiCamSurveillanceSingleton:
    _instance = None
    _model = None  # Thêm instance YOLOv5 chung
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                # Tạo file config tạm
                import json
                config_path = 'yt_cameras.json'
                config_dict = {'cameras': [c.__dict__ for c in CAMERA_CONFIGS]}
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f)
                cls._instance = main_V4.MultiCameraSurveillanceSystem(config_file=config_path, batch_size=4)
                # Khởi tạo model YOLOv5 chung
                cls._model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
                cls._model.eval()
                if torch.cuda.is_available():
                    cls._model.to('cuda')
                threading.Thread(target=cls._instance.start, daemon=True).start()
            return cls._instance

    @classmethod
    def get_model(cls):
        with cls._lock:
            if cls._model is None:
                cls._model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
                cls._model.eval()
                if torch.cuda.is_available():
                    cls._model.to('cuda')
            return cls._model

# --- Context manager để quản lý file tạm ---
@contextmanager
def temp_file(suffix='.mp4'):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        yield tmp
    finally:
        tmp.close()
        try:
            os.remove(tmp.name)
        except Exception as e:
            logger.error(f"Failed to delete temp file {tmp.name}: {e}")

# --- Streaming views ---
def gen_surveillance_stream(camera_id):
    system = MultiCamSurveillanceSingleton.get_instance()
    while True:
        frame = None
        with system.frame_cache_lock:
            frame = system.frame_cache.get(camera_id)
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            # Thêm status text lên frame trống
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            status = f"Camera {camera_id}: Waiting for frame"
            cv2.putText(blank, status, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)

def stream_cam1(request):
    return StreamingHttpResponse(gen_surveillance_stream('cam1'), content_type='multipart/x-mixed-replace; boundary=frame')

def stream_cam2(request):
    return StreamingHttpResponse(gen_surveillance_stream('cam2'), content_type='multipart/x-mixed-replace; boundary=frame')

def multi_youtube(request):
    return render(request, 'tracking/multi_youtube.html')

# --- Upload detect (ảnh) ---
@csrf_exempt
def detect_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file: InMemoryUploadedFile = request.FILES['image']
        image = Image.open(image_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        try:
            # Sử dụng model chung
            model = MultiCamSurveillanceSingleton.get_model()
            config = CAMERA_CONFIGS[0]  # Sử dụng cấu hình của cam1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model([rgb_frame], size=640)
            predictions = results.pred[0]
            # Extract detections
            detections = []
            for *xyxy, conf, cls in predictions:
                if int(cls) == 0 and conf > config.confidence_threshold:
                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': float(conf),
                        'area': width * height,
                        'height_pixels': height
                    })
            # Tracking và giám sát khoảng cách
            tracker = main_V4.PersonTracker(camera_id='img', config=config)
            tracker.update_tracks(detections)
            close_pairs = tracker.draw_tracks(frame)  # Vẽ tracks và lấy close_pairs
            # Thêm thông tin vi phạm khoảng cách
            violations = [
                {'id1': id1, 'id2': id2, 'distance': float(distance)}
                for id1, id2, distance in close_pairs
            ]
            _, jpeg = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(jpeg.tobytes()).decode()
            return JsonResponse({
                'result': img_str,
                'violations': violations,
                'statistics': tracker.get_statistics()
            })
        except Exception as e:
            logger.error(f"Detect image failed: {e}\n{traceback.format_exc()}")
            return JsonResponse({'error': f'Detect failed: {str(e)}'}, status=500)
    return JsonResponse({'error': 'No image uploaded'}, status=400)

# --- Upload detect (video) ---
@csrf_exempt
def detect_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file: InMemoryUploadedFile = request.FILES['video']
        # Sử dụng context manager cho file tạm
        with temp_file(suffix='.mp4') as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
            try:
                # Sử dụng model chung
                model = MultiCamSurveillanceSingleton.get_model()
                config = CAMERA_CONFIGS[0]  # Sử dụng cấu hình của cam1
                tracker = main_V4.PersonTracker(camera_id='vid', config=config)
                cap = cv2.VideoCapture(tmp_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_w, out_h = width, height
                # Tạo file video output
                with temp_file(suffix='.mp4') as out_tmp:
                    out = cv2.VideoWriter(out_tmp.name, fourcc, fps, (out_w, out_h))
                    violations = []  # Lưu danh sách vi phạm
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model([rgb_frame], size=640)
                        predictions = results.pred[0]
                        detections = []
                        for *xyxy, conf, cls in predictions:
                            if int(cls) == 0 and conf > config.confidence_threshold:
                                x1, y1, x2, y2 = map(int, xyxy)
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                width = x2 - x1
                                height = y2 - y1
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'center': (center_x, center_y),
                                    'confidence': float(conf),
                                    'area': width * height,
                                    'height_pixels': height
                                })
                        tracker.update_tracks(detections)
                        close_pairs = tracker.draw_tracks(frame)
                        # Ghi chú vi phạm lên frame
                        if close_pairs:
                            violation_text = f"Violations: {len(close_pairs)}"
                            cv2.putText(frame, violation_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            violations.extend([
                                {'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                 'id1': id1, 'id2': id2, 'distance': float(distance)}
                                for id1, id2, distance in close_pairs
                            ])
                        out.write(frame)
                    cap.release()
                    out.release()
                    out_tmp.seek(0)
                    video_bytes = out_tmp.read()
                response = HttpResponse(video_bytes, content_type='video/mp4')
                response['Content-Disposition'] = 'attachment; filename="detected.mp4"'
                # Có thể trả về thêm thông tin vi phạm trong header hoặc file riêng
                response['X-Violations'] = json.dumps(violations)
                return response
            except Exception as e:
                logger.error(f"Detect video failed: {e}\n{traceback.format_exc()}")
                return JsonResponse({'error': f'Detect failed: {str(e)}'}, status=500)
    return JsonResponse({'error': 'No video uploaded'}, status=400)

# --- Trang upload detect UI ---
def detect_upload(request):
    return render(request, 'tracking/detect_upload.html')

# --- Webcam detect UI ---
def webcam(request):
    return render(request, 'tracking/webcam.html')

# --- Trang index hoặc multi-youtube ---
def index(request):
    return render(request, 'tracking/index.html')