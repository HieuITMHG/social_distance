from django.http import StreamingHttpResponse
import cv2
import numpy as np

from django.shortcuts import render
from .distance import PersonTracker

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
import base64
from PIL import Image
import io
import tempfile

# Khởi tạo tracker toàn cục cho webcam
webcam_tracker = PersonTracker(model_name='yolov5s', confidence_threshold=0.5)

def index(request):
    return render(request, 'tracking/index.html')

def gen():
    cap = cv2.VideoCapture(0)
    tracker = webcam_tracker
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections, _ = tracker.detect_persons(frame)
        tracker.update_tracks(detections)
        tracker.draw_tracks(frame)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    cap.release()

def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def detect_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file: InMemoryUploadedFile = request.FILES['image']
        image = Image.open(image_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        tracker = PersonTracker(model_name='yolov5s', confidence_threshold=0.5)
        detections, _ = tracker.detect_persons(frame)
        tracker.update_tracks(detections)
        tracker.draw_tracks(frame)
        # Chuyển ảnh kết quả sang base64
        buffer = io.BytesIO()
        result_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(result_img_rgb).save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return JsonResponse({'result': img_str})
    return JsonResponse({'error': 'No image uploaded'}, status=400)

@csrf_exempt
def detect_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file: InMemoryUploadedFile = request.FILES['video']
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        # Xử lý video bằng PersonTracker
        tracker = PersonTracker(model_name='yolov5s', confidence_threshold=0.5)
        # Đọc video, detect, vẽ, trả về bytes mp4
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
                detections, _ = tracker.detect_persons(frame)
                tracker.update_tracks(detections)
                tracker.draw_tracks(frame)
                out.write(frame)
            cap.release()
            out.release()
            out_tmp.seek(0)
            video_bytes = out_tmp.read()
        from django.http import HttpResponse
        response = HttpResponse(video_bytes, content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename="detected.mp4"'
        return response
    return JsonResponse({'error': 'No video uploaded'}, status=400)

def webcam_detect(request):
    return render(request, 'tracking/webcam_detect.html')

def upload_detect(request):
    return render(request, 'tracking/upload_detect.html')