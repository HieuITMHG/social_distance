import torch
import cv2
import numpy as np
import tempfile

# Load model YOLOv5s (nhanh hơn)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detect_and_draw(frame, max_size=900):
    # Nếu là PIL Image thì convert sang numpy
    if hasattr(frame, 'convert'):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = frame.shape[:2]
    # Chuyển frame BGR (OpenCV) sang RGB (YOLOv5)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    # Lấy kết quả
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) != 0:
            continue  # chỉ vẽ class person
        x1, y1, x2, y2 = map(int, box)
        # Vẽ bounding box không label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    # Nếu ảnh quá lớn thì resize về max_size
    h, w = frame.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return frame


def detect_and_draw_video(video_path, max_size=480):
    """
    Detect người trên video, trả về bytes video đã detect (mp4)
    video_path: đường dẫn file video upload
    max_size: resize frame về kích thước này để tăng tốc
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fps = min(fps, 20)  # Giảm FPS tối đa 20 để tăng tốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Resize về max_size
    scale = min(max_size / max(width, height), 1.0)
    out_w, out_h = int(width * scale), int(height * scale)
    # Tạo file tạm để lưu video detect
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        out = cv2.VideoWriter(tmp.name, fourcc, fps, (out_w, out_h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize trước khi detect
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            result_frame = detect_and_draw(frame, max_size=max_size)
            out.write(result_frame)
        cap.release()
        out.release()
        tmp.seek(0)
        video_bytes = tmp.read()
    return video_bytes