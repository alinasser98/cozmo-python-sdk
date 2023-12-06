import time
import cozmo
import asyncio
from cozmo.util import degrees
from cozmo.objects import LightCube1Id

# Set the initial head angle (in degrees)
initial_head_angle = 22

def capture_and_save_image(robot, image_count):
    # Set the head angle
    robot.set_head_angle(degrees(initial_head_angle)).wait_for_completed()
    
    # Take a picture and save with unique name
    image = robot.world.latest_image.raw_image
    image.save(f"Cheap AI art mountain lake trees{image_count}.png", 'PNG')

def cozmo_program(robot: cozmo.robot.Robot):
    image_count = 0
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True

    # Connecting to cube
    robot.world.connect_to_cubes()
    cube1 = robot.world.get_light_cube(LightCube1Id)
    if cube1:
        cube1.set_lights(cozmo.lights.green_light)
    else:
        cozmo.logger.warning("Lightcube for Cozmo is not connected maybe the battery is dead.")

    # Function to handle cube taps
    def on_cube_tap1(evt, **kwargs):
        nonlocal image_count
        image_count += 1
        asyncio.ensure_future(capture_and_save_image(robot, image_count))

    # Event handler for cube 1 tap
    cube1.add_event_handler(cozmo.objects.EvtObjectTapped, on_cube_tap1)

    # Keep the program running until stopped
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program stopped by user.")

# Run the program
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
