from tqdm import tqdm

from data_stream import StreamReceiver
from simulation.webots.controllers.ardupilot_vehicle_controller.drone_data import DroneData

import cv2
from ultralytics import YOLO
import random
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.deep.extractor import Extractor
from deep_sort.deep_sort.deep.configuration import ResNetConfiguration
from deep_sort.deep_sort.deep.weights import RESNET18_WEIGHTS


host = "192.168.0.107"
port = 5588
stream_receiver = StreamReceiver(host, port)

model = YOLO("yolov8n.pt")

detection_threshold = 0.01

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

data = stream_receiver.get_data()
camera = DroneData.from_json(data).camera


cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*"MP4V"),
    camera.fps,
    (camera.width, camera.height)
)

frame_count = camera.fps * 10

progress_bar = tqdm(total=frame_count, desc="Processing frames", unit="frame")

count = 0

while count < frame_count:
    data = stream_receiver.get_data()

    drone_data = DroneData.from_json(data)

    camera_frame = drone_data.camera.frame

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

    count += 1

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    cap_out.write(print_frame)
    progress_bar.update(1)

cap_out.release()
progress_bar.close()
