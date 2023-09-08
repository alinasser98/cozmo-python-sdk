import socket
import cozmo
from cozmo.util import degrees, distance_mm

# Define Cozmo's name
COZMO_NAME = "robobuddy"

def parse_and_execute_movement(robot: cozmo.robot.Robot, message: str):
    _, FB, LR, distX, distY = message.split(";")

    if FB == 'F' and int(distX) != 0:
        robot.drive_straight(distance_mm(int(distX)), cozmo.util.speed_mmps(150)).wait_for_completed()
    elif FB == 'B' and int(distX) != 0:
        robot.turn_in_place(degrees(180)).wait_for_completed()
        robot.drive_straight(distance_mm(int(distX)), cozmo.util.speed_mmps(150)).wait_for_completed()

    if LR == 'L' and int(distY) != 0:
        robot.turn_in_place(degrees(90)).wait_for_completed()
        robot.drive_straight(distance_mm(int(distY)), cozmo.util.speed_mmps(150)).wait_for_completed()
    elif LR == 'R' and int(distY) != 0:
        robot.turn_in_place(degrees(-90)).wait_for_completed()
        robot.drive_straight(distance_mm(int(distY)), cozmo.util.speed_mmps(150)).wait_for_completed()

def parse_and_execute_head_tractor(robot: cozmo.robot.Robot, message: str):
    _, headValue, tractorValue = message.split(";")
    robot.set_head_angle(degrees(float(headValue))).wait_for_completed()
    robot.move_lift(float(tractorValue))

def cozmo_control_client(robot: cozmo.robot.Robot):
    SERVER_IP, SERVER_PORT = '10.0.1.10', 5000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, SERVER_PORT))
        print("Connected to server. Waiting for commands...")
        
        while True:
            data = s.recv(4096).decode('utf-8')  # Receive the message from the chat server
            
            # If no data received, server might have closed the connection. Break the loop.
            if not data:
                print("Connection closed by server.")
                break

            parts = data.split(';')
            if parts[0] != COZMO_NAME:  # Check if the message is for our Cozmo
                continue

            # Check for the type of command based on the number of parts in the message
            if len(parts) == 5:  # Movement command
                parse_and_execute_movement(robot, data)
            elif len(parts) == 3:  # Head and tractor command
                parse_and_execute_head_tractor(robot, data)
            else:
                print("Invalid command received!")

cozmo.run_program(cozmo_control_client, use_viewer=False, force_viewer_on_top=False)
