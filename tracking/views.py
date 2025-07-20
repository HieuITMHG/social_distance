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
from django.conf import settings
from tracking import config_system
from tracking import main_V4
import logging
import traceback
import torch

logger = logging.getLogger("detect_upload")

# --- Camera configs: 3 YouTube + 1 webcam ---
CAMERA_CONFIGS = [
    main_V4.CameraConfig(camera_id='webcam', source='0', position='Webcam',
                 enable_recording=config_system.ENABLE_RECORDING,
                 recording_path=config_system.RECORDING_PATH,
                 confidence_threshold=config_system.CONFIDENCE_THRESHOLD,
                 social_distance_threshold=config_system.SOCIAL_DISTANCE_THRESHOLD,
                 warning_duration=config_system.WARNING_DURATION,
                 loop_video=False),
    main_V4.CameraConfig(camera_id='yt1', source=config_system.YOUTUBE_URL_1, position='YouTube 1',
                 enable_recording=config_system.ENABLE_RECORDING,
                 recording_path=config_system.RECORDING_PATH,
                 confidence_threshold=config_system.CONFIDENCE_THRESHOLD,
                 social_distance_threshold=config_system.SOCIAL_DISTANCE_THRESHOLD,
                 warning_duration=config_system.WARNING_DURATION,
                 loop_video=True),
    main_V4.CameraConfig(camera_id='yt2', source=config_system.YOUTUBE_URL_2, position='YouTube 2',
                 enable_recording=config_system.ENABLE_RECORDING,
                 recording_path=config_system.RECORDING_PATH,
                 confidence_threshold=config_system.CONFIDENCE_THRESHOLD,
                 social_distance_threshold=config_system.SOCIAL_DISTANCE_THRESHOLD,
                 warning_duration=config_system.WARNING_DURATION,
                 loop_video=True),
    main_V4.CameraConfig(camera_id='yt3', source=config_system.YOUTUBE_URL_3, position='YouTube 3',
                 enable_recording=config_system.ENABLE_RECORDING,
                 recording_path=config_system.RECORDING_PATH,
                 confidence_threshold=config_system.CONFIDENCE_THRESHOLD,
                 social_distance_threshold=config_system.SOCIAL_DISTANCE_THRESHOLD,
                 warning_duration=config_system.WARNING_DURATION,
                 loop_video=True),
]

# --- Singleton sử dụng configs trên ---
class MultiCamSurveillanceSingleton:
    _instance = None
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
                threading.Thread(target=cls._instance.start, daemon=True).start()
            return cls._instance

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
            blank = (np.zeros((480, 640, 3), dtype=np.uint8))
            _, jpeg = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)

def stream_webcam(request):
    return StreamingHttpResponse(gen_surveillance_stream('webcam'), content_type='multipart/x-mixed-replace; boundary=frame')

def stream_yt1(request):
    return StreamingHttpResponse(gen_surveillance_stream('yt1'), content_type='multipart/x-mixed-replace; boundary=frame')

def stream_yt2(request):
    return StreamingHttpResponse(gen_surveillance_stream('yt2'), content_type='multipart/x-mixed-replace; boundary=frame')

def stream_yt3(request):
    return StreamingHttpResponse(gen_surveillance_stream('yt3'), content_type='multipart/x-mixed-replace; boundary=frame')

def multi_youtube(request):
    return render(request, 'tracking/multi_youtube.html')

# --- Upload detect (ảnh/video) ---
@csrf_exempt
def detect_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file: InMemoryUploadedFile = request.FILES['image']
        image = Image.open(image_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        try:
            # Load model YOLOv5m
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
            model.eval()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model([rgb_frame], size=640)
            predictions = results.pred[0]
            # Extract detections
            detections = []
            for *xyxy, conf, cls in predictions:
                if int(cls) == 0 and conf > 0.5:
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
            # Tracking
            config = main_V4.CameraConfig(camera_id='img', source='', position='Upload')
            tracker = main_V4.PersonTracker(camera_id='img', config=config)
            tracker.update_tracks(detections)
            tracker.draw_tracks(frame)
            _, jpeg = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(jpeg.tobytes()).decode()
            return JsonResponse({'result': img_str})
        except Exception as e:
            logger.error(f"Detect image failed: {e}\n{traceback.format_exc()}")
            return JsonResponse({'error': f'Detect failed: {str(e)}'}, status=500)
    return JsonResponse({'error': 'No image uploaded'}, status=400)

@csrf_exempt
def detect_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file: InMemoryUploadedFile = request.FILES['video']
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            # Load model YOLOv5m
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
            model.eval()
            config = main_V4.CameraConfig(camera_id='vid', source='', position='Upload')
            tracker = main_V4.PersonTracker(camera_id='vid', config=config)
            cap = cv2.VideoCapture(tmp_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w, out_h = width, height
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as out_tmp:
                out = cv2.VideoWriter(out_tmp.name, fourcc, fps, (out_w, out_h))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model([rgb_frame], size=640)
                    predictions = results.pred[0]
                    detections = []
                    for *xyxy, conf, cls in predictions:
                        if int(cls) == 0 and conf > 0.5:
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
                    tracker.draw_tracks(frame)
                    out.write(frame)
                cap.release()
                out.release()
                out_tmp.seek(0)
                video_bytes = out_tmp.read()
            response = HttpResponse(video_bytes, content_type='video/mp4')
            response['Content-Disposition'] = 'attachment; filename="detected.mp4"'
            return response
        except Exception as e:
            logger.error(f"Detect video failed: {e}\n{traceback.format_exc()}")
            return JsonResponse({'error': f'Detect failed: {str(e)}'}, status=500)
        finally:
            os.remove(tmp_path)
    return JsonResponse({'error': 'No video uploaded'}, status=400)

# --- Trang upload detect UI ---
def detect_upload(request):
    return render(request, 'tracking/detect_upload.html')

# --- Webcam detect UI (nếu cần) ---
def webcam(request):
    return render(request, 'tracking/webcam.html')

# --- Trang index hoặc multi-youtube ---
def index(request):
    return render(request, 'tracking/index.html')