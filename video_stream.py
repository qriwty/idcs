import socket
import struct
import numpy as np


class VideoStreamClient:
    def __init__(self, host, port, channels=3, data_type=np.uint8):
        self.host = host
        self.port = port

        self.channels = channels
        self.data_type = data_type

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

        self.video_frames_generator = self._receive_video_frames()

    def _read_exact_bytes(self, num_bytes):
        data = b''

        while len(data) < num_bytes:
            chunk = self.client_socket.recv(num_bytes - len(data))

            if not chunk:
                raise ConnectionError()

            data += chunk

        return data

    def _receive_video_frames(self):
        while True:
            size_data = self.client_socket.recv(4)

            if not size_data:
                break

            width, height = struct.unpack('HH', size_data)

            bytes_per_pixel = self.data_type().nbytes
            img_data_size = width * height * self.channels * bytes_per_pixel

            img_data = self._read_exact_bytes(img_data_size)

            img = np.frombuffer(img_data, dtype=self.data_type).reshape((height, width, self.channels))

            yield img

    def current_frame(self):
        return next(self.video_frames_generator)