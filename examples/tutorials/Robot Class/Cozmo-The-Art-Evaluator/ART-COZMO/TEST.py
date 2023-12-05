import time
import cozmo
import asyncio
from cozmo.util import degrees
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image
import tensorflow as tf

# Load your TensorFlow model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Art Information Database
art_database = {
    # Your provided art database entries
    "Cheap_AI_art_mountain_lake_trees": {
        "name": "I am not sure what the name is but it seems very serene",
        # ... (other details as provided)
    },
    # ... (other artworks)
}

# Function to process image through the model
def process_image(image):
    # Preprocess the image as required by your model
    # Predict using the model
    # Return the identified art name
    pass

# Function to get art information
def get_art_info(art_name):
    default_info = {
        'name': 'Unknown',
        'remark': 'No information available.',
        'artist': 'Unknown',
        'year': 'Unknown',
        'style': 'Unknown',
        'description': 'No description available.',
        'appraisal': 'No appraisal value available.'
    }
    return art_database.get(art_name, default_info)

# Function to capture and process an image
async def capture_and_process_image(robot):
    robot.set_head_angle(degrees(0)).wait_for_completed()
    image = robot.world.latest_image.raw_image
    art_name = process_image(image)
    return art_name

# Function to handle user response (Yes/No cubes)
async def handle_user_response(robot, yes_cube, no_cube):
    yes_tapped = False
    no_tapped = False

    def on_yes_cube_tap(evt, **kwargs):
        nonlocal yes_tapped
        yes_tapped = True

    def on_no_cube_tap(evt, **kwargs):
        nonlocal no_tapped
        no_tapped = True

    yes_cube.add_event_handler(cozmo.objects.EvtObjectTapped, on_yes_cube_tap)
    no_cube.add_event_handler(cozmo.objects.EvtObjectTapped, on_no_cube_tap)

    while not (yes_tapped or no_tapped):
        await asyncio.sleep(0.1)

    if yes_tapped:
        robot.say_text("Thank you for your business!").wait_for_completed()
    elif no_tapped:
        robot.say_text("Your loss, I've got the best offers around!").wait_for_completed()

# Main Cozmo program
def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.world.connect_to_cubes()

    # Set up cubes
    yes_cube = robot.world.get_light_cube(LightCube2Id)  # Cube for 'Yes' response
    no_cube = robot.world.get_light_cube(LightCube3Id)   # Cube for 'No' response
    camera_cube = robot.world.get_light_cube(LightCube1Id)  # Cube to start camera

    if camera_cube is not None:
        camera_cube.set_lights(cozmo.lights.green_light)
    else:
        cozmo.logger.warning("Cube 1 not connected.")

    # Function to handle camera cube tap
    async def on_camera_cube_tap(evt, **kwargs):
        art_name = await capture_and_process_image(robot)
        art_info = get_art_info(art_name)

        robot.say_text(f"Name of the artwork: {art_info['name']}").wait_for_completed()
        robot.say_text(f"Remark: {art_info['remark']}").wait_for_completed()
        robot.say_text(f"Artist: {art_info['artist']}").wait_for_completed()
        robot.say_text(f"Year: {art_info['year']}").wait_for_completed()
        robot.say_text(f"Style: {art_info['style']}").wait_for_completed()
        robot.say_text(f"Description: {art_info['description']}").wait_for_completed()
        robot.say_text(f"Appraisal Value: {art_info['appraisal']}").wait_for_completed()

        await handle_user_response(robot, yes_cube, no_cube)

    # Add event handler for camera cube tap
    camera_cube.add_event_handler(cozmo.objects.EvtObjectTapped, on_camera_cube_tap)

    while True:
        time.sleep(1)

# Run the program
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
