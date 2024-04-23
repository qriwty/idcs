from data_stream import StreamReceiver

from simulation.webots.controllers.ardupilot_vehicle_controller.drone_data import RangefinderData, CameraData, FDMData, GimbalAxisData, GimbalData, DroneData
import cv2


host = "192.168.0.107"
port = 5588
stream_receiver = StreamReceiver(host, port)


frame_count = 0
for data in stream_receiver:
    # print("Data received", data)
    if "camera" in data:
        camera = CameraData.from_dict(data["camera"])
        frame = camera.frame
        if frame is not None:
            # print(f"Displaying frame {frame_count}")
            cv2.imshow("Frame", frame)
            frame_count += 1
        else:
            print("Frame is None, skipping...")
    else:
        print("No 'camera' key in data")

    key = cv2.waitKey(25)
    if key & 0xFF == ord("q"):
        print("Exiting on user request")
        break
    elif key != -1:
        print(f"Key pressed: {key}")
