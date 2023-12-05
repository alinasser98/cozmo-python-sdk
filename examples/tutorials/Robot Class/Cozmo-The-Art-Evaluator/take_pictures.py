import time
import cozmo
import asyncio
from cozmo.util import degrees
from cozmo.objects import LightCube1Id
#=========================================================================================================>
#========================================================================> Set the head angle (in degrees)
desired_head_angle = 0

def capture_and_save_image(robot):
    #____________________________________________ Set the head angle
    robot.set_head_angle(degrees(desired_head_angle)).wait_for_completed()
    
    #____________________________________________ Take a picture and save
    image = robot.world.latest_image.raw_image
    image.save(f"{time.time()}.png", 'PNG')


#=========================================================================================================>
#========================================================================>
def cozmo_program(robot: cozmo.robot.Robot):
    #____________________________________________ Turning on image stream and colours
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    
    #____________________________________________ Connecting to cube
    robot.world.connect_to_cubes()
    cube1 = robot.world.get_light_cube(LightCube1Id)
    if cube1 is not None:
        cube1.set_lights(cozmo.lights.green_light)
    else:
        cozmo.logger.warning("Cozmo is not connected to a LightCube1Id cube - check the battery.")

    #____________________________________________ Function to handle cube taps () asyncio
    def on_cube_tap1(evt, **kwargs):
        asyncio.ensure_future(capture_and_save_image(robot))

    #____________________________________________ Event handler for cube 1 tap
    cube1.add_event_handler(cozmo.objects.EvtObjectTapped, on_cube_tap1)

    #____________________________________________ Keep the program running until the user stops it
    while True:
        time.sleep(1)

#=========================================================================================================>
#========================================================================>  Run the program
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
