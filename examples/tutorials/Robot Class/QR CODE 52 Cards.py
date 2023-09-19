# Imports
import cozmo
import cv2
from pyzbar.pyzbar import decode
import qrcode
import socket

# Constants
COZMO_NAME = "robobuddy"
SERVER_IP, SERVER_PORT = '10.0.1.10', 5000

# Cozmo QR Code Scanning + Blackjack Logic
def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    total_value = 0
    cards_detected = []

    while True:
        image = robot.world.latest_image.raw_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_qrcodes = decode(gray)
        for qr in detected_qrcodes:
            card = qr.data.decode('utf-8')
            if card not in cards_detected:
                cards_detected.append(card)
                value, suit = card.split("_")
                if value in ["Jack", "Queen", "King"]:
                    total_value += 10
                elif value == "Ace":
                    total_value += 11 if total_value <= 10 else 1
                else:
                    total_value += int(value)
                robot.say_text(f"{value} of {suit}, hand value is {total_value}").wait_for_completed()
                # Send to server
                send_data_to_server(";".join([COZMO_NAME] + cards_detected))
                # Blackjack Logic
                if total_value < 17:
                    robot.say_text("I choose to HIT!").wait_for_completed()
                    robot.drive_straight(distance_mm(50), cozmo.util.speed_mmps(50)).wait_for_completed()
                else:
                    robot.say_text("I choose to STAY!").wait_for_completed()
                    robot.turn_in_place(degrees(180)).wait_for_completed()
                    return

# Networking / Server code
def send_data_to_server(message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, SERVER_PORT))
        s.sendall(message.encode('utf-8'))

# Run the Cozmo program
cozmo.run_program(cozmo_program)
