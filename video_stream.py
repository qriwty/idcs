import socket
import json
import time


class StreamReceiver:
    def __init__(self, host, port, max_retries=5, retry_delay=5):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client_socket = None
        self.connect()

    def connect(self):
        """Create and connect the client socket to the server with retries."""
        attempts = 0
        while attempts < self.max_retries:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                print("Connected to server at {}:{}".format(self.host, self.port))
                return
            except socket.error as e:
                print("Failed to connect to {}:{}. Reason: {}. Retrying in {} seconds...".format(
                    self.host, self.port, e, self.retry_delay))
                attempts += 1
                time.sleep(self.retry_delay)
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None

        raise ConnectionError("Failed to connect after {} attempts".format(self.max_retries))

    def receive_data(self):
        """Receive data from the server and yield complete JSON objects."""
        buffer = ""
        while True:
            try:
                data = self.client_socket.recv(1024).decode('utf-8')
                if not data:
                    # No more data received, attempt to reconnect
                    print("Connection lost. Attempting to reconnect...")
                    self.connect()
                    continue

                buffer += data

                while "\n" in buffer:
                    json_object, _, buffer = buffer.partition("\n")
                    if json_object:
                        try:
                            json_data = json.loads(json_object)
                            yield json_data
                        except json.JSONDecodeError:
                            continue

            except socket.error as e:
                print("Socket error during data reception:", e)
                print("Attempting to reconnect...")

                self.connect()

    def disconnect(self):
        """Disconnect the client socket."""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            print("Disconnected from server.")

    def get_data(self):
        return next(self)

    def __iter__(self):
        """Allow the StreamReceiver to be an iterable."""
        return self.receive_data()
