from tqdm import tqdm

from video_stream import VideoStreamClient
import cv2
import numpy as np
from ultralytics import YOLO
import random
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.deep.extractor import Extractor
from deep_sort.deep_sort.deep.configuration import ResNetConfiguration
from deep_sort.deep_sort.deep.weights import RESNET18_WEIGHTS


camera_stream = VideoStreamClient("192.168.0.107", 5588, channels=3, data_type=np.uint8)
depth_stream = VideoStreamClient("192.168.0.107", 5599, channels=1, data_type=np.uint8)


model = YOLO("yolov8n.pt")

detection_threshold = 0.1

resnet = ResNetConfiguration(
    base="resnet18",
    weights_path=RESNET18_WEIGHTS,
    use_cuda=True
)
extractor = Extractor(model=resnet, batch_size=4)

tracker = Tracker(
    feature_extractor=extractor
)

colors = [(
    random.randint(0, 255),
    random.randint(0, 255),
    random.randint(0, 255)) for j in range(10)
]

video_out_path = "out_running.mp4"

frame = camera_stream.current_frame()

cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*"MP4V"),
    24,
    (frame.shape[1], frame.shape[0])
)

frame_count = 24 * 60

progress_bar = tqdm(total=frame_count, desc="Processing frames", unit="frame")

count = 0

while count < frame_count:
    camera_frame = camera_stream.current_frame()
    depth_frame = depth_stream.current_frame()

    frame = camera_frame

    results = model.predict(
        camera_frame,
        verbose=False
    )
    result = results[0]

    detections = []
    for r in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        class_id = int(class_id)

        if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score, class_id])

    tracker.update(frame, detections)

    print_frame = frame.copy()

    for track in tracker.tracks:
        x1, y1, x2, y2 = track.to_tlbr()
        track_id = track.track_id
        class_id = track.class_id

        color = colors[track_id % len(colors)]

        cv2.rectangle(
            print_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            3
        )
        cv2.putText(
            print_frame,
            f"ID: {track_id} | Class: {model.names[class_id]}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[track_id % len(colors)],
            2
        )

    cv2.imshow("camera_frame", print_frame)

    cv2.imshow("depth_frame", depth_frame)

    count += 1

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    cap_out.write(print_frame)
    progress_bar.update(1)

cap_out.release()
progress_bar.close()
