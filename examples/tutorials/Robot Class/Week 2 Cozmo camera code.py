import cozmo
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image
import asyncio
from cozmo.util import degrees, distance_mm
import _thread
import time

def on_object_tapped(self, event, *, obj, tap_count, tap_duration, **kw):
    robot.say_text("The cube was tapped").wait_for_completed()
    return

def moveAround(robot, speed1, speed2, seconds):
    robot.drive_wheels(speed1, speed2, None, None, seconds)
    return

def cozmo_program(robot: cozmo.robot.Robot):
    
    success = True
    
    # see what Cozmo sees
    robot.camera.image_stream_enabled = True
    
    # connect to cubes (in case Cozmo was disconnected from the cubes)
    robot.world.connect_to_cubes()
    
    # identify cubes
    cube1 = robot.world.get_light_cube(LightCube1Id)  
    cube2 = robot.world.get_light_cube(LightCube2Id)  
    cube3 = robot.world.get_light_cube(LightCube3Id)  

    if cube1 is not None:
        cube1.set_lights(cozmo.lights.red_light)
    else:
        cozmo.logger.warning("Cozmo is not connected to a LightCube1Id cube - check the battery.")

    if cube2 is not None:
        cube2.set_lights(cozmo.lights.green_light)
    else:
        cozmo.logger.warning("Cozmo is not connected to a LightCube2Id cube - check the battery.")

    if cube3 is not None:
        cube3.set_lights(cozmo.lights.blue_light)
    else:
        cozmo.logger.warning("Cozmo is not connected to a LightCube3Id cube - check the battery.") 

    # have the user tap each of the cubes, in order
    try:
        cube1.wait_for_tap(timeout=60)
    except asyncio.TimeoutError:
        robot.say_text("The red cube was not tapped").wait_for_completed()
        success = False
    finally:
        cube1.set_lights_off()
        if (success):
            moveAround(robot, 800, 650, 10)# Move straight for 2 seconds after the cube is tapped
            robot.say_text("My owner is Ali Nasser").wait_for_completed()

        else:
            robot.say_text("You didn't tap the cube properly.").wait_for_completed()
        success = True
            
    return

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
