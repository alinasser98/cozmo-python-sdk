#import the cozmo and image libraries
import cozmo
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image


#import libraries for movement and asynchronous behavior
import asyncio
from cozmo.util import degrees, distance_mm

#import these libraries when needed for threads
import _thread
import time


def on_object_tapped(self, event, *, obj, tap_count, tap_duration, **kw):
	robot.say_text("The cube was tapped").wait_for_completed()
	return

def cozmo_program(robot: cozmo.robot.Robot):
	
	success = True
	
	#see what Cozmo sees
	robot.camera.image_stream_enabled = True
	
	#connect to cubes (in case Cozmo was disconnected from the cubes)
	robot.world.connect_to_cubes()
	
	#identify cubes
	cube1 = robot.world.get_light_cube(LightCube1Id)  # looks like a paperclip
	cube2 = robot.world.get_light_cube(LightCube2Id)  # looks like a lamp / heart
	cube3 = robot.world.get_light_cube(LightCube3Id)  # looks like the letters 'ab' over 'T'

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

	
  
	#have the user tap each of the cubes, in order
	try:
		robot.say_text("Tap the red cube and make me say something.").wait_for_completed()
		cube1.wait_for_tap(timeout=10)
	except asyncio.TimeoutError:
		robot.say_text("The red cube was not tapped").wait_for_completed()
		success = False
	finally:
		cube1.set_lights_off()
		if (success):
			robot.say_text("Thank you for your service.").wait_for_completed()
		else:
			robot.say_text("You didn't tap the cube properly.").wait_for_completed()
		success = True

	
	try:
		robot.say_text("Tap the green cube so I can take a picture.").wait_for_completed()
		cube2.wait_for_tap(timeout=10)
	except asyncio.TimeoutError:
		robot.say_text("The green cube was not tapped").wait_for_completed()
		success = False
	finally:
		cube2.set_lights_off()
		if (success):
			robot.say_text("I see that you were paying attention.  I will take a picture.").wait_for_completed()
		else:
			robot.say_text("Do you know how to tap a cube? I will take a picture anyway.").wait_for_completed()
		success = True	
		
		new_im = robot.world.wait_for(cozmo.world.EvtNewCameraImage)
		new_im.image.raw_image.show()
	
		#save the raw image as a bmp file
		img_latest = robot.world.latest_image.raw_image
		img_convert = img_latest.convert('L')
		img_convert.save("aPhoto.bmp")
	
		#save the raw image data as a png file, named imageName
		imageName = "myPhoto.png"
		img = Image.open("aPhoto.bmp")
		width, height = img.size
		new_img = img.resize( (width, height) )
		new_img.save( imageName, 'png')	
	
	try:
		robot.say_text("Tap the blue cube and make me do something.").wait_for_completed()
		cube3.wait_for_tap(timeout=10)
	except asyncio.TimeoutError:
		robot.say_text("The blue cube was not tapped").wait_for_completed()
		success = False
	finally:
		cube3.set_lights_off()
		cube = robot.world.wait_for_observed_light_cube()
		
		if (success):
			robot.say_text("Well done! I will pop a wheelie.").wait_for_completed()
		else:
			robot.say_text("Do you know how to tap a cube? I will pop a wheelie.").wait_for_completed()
		success = True	

		action = robot.pop_a_wheelie(cube, num_retries=2)
		action.wait_for_completed()
		
	robot.say_text("This is awkward.  I didn't think this through.  Help me.").wait_for_completed()
	
	return

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)