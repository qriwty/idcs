from data_stream import StreamReceiver

from simulation.webots.controllers.ardupilot_vehicle_controller.drone_data import RangefinderData, CameraData, FDMData, GimbalAxisData, GimbalData, DroneData
import cv2


host = "192.168.0.107"
port = 5588
stream_receiver = StreamReceiver(host, port)


frame_count = 0
while True:
    data = stream_receiver.get_data()
    drone_data = DroneData.from_json(data)
    camera_frame = drone_data.camera.frame

    print(f"Frame received: {frame_count}")

    cv2.imshow("Frame", camera_frame)
    frame_count += 1

    key = cv2.waitKey(25)
    if key & 0xFF == ord("q"):
        print("Exiting on user request")
        break
    elif key != -1:
        print(f"Key pressed: {key}")
