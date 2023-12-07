import time
import cozmo
import asyncio
from cozmo.util import degrees
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image
import tensorflow as tf
import numpy as np


# Load your TensorFlow model
model = tf.keras.models.load_model('examples/tutorials/Robot Class/Cozmo-The-Art-Evaluator/ART-COZMO/Art_Eval_For_Cozmo_The_Evaluator.keras')

# Art Information Database
art_database = {
    "Edvard-Munch-The-Scream-1893": {
        "name": " This is Edvard Munch's The Scream",
        "remark": "This is a very famous painting and I think it will sell for a lot of money",
        "artist": "The artist is Edvard Munch",
        "year": "Eighteen ninety three",
        "style": "The style is Expressionism",
        "description": "This is one of Munch's four versions of 'The Scream', it symbolizes modern existential angst and has an iconic status in art history.",
        "appraisal": "I want to offer you one hundred and fifty million dollars"
    },
    "Georges-Seurat-A-Sunday-Afternoon": {
        "name": " This is Georges Seurat's A Sunday Afternoon on the Island of La Grande Jatte",
        "remark": "This is a very famous painting and I think it will sell for a ton of money",
        "artist": "The artist is Georges Seurat",
        "year": "Eighteen eighty five",
        "style": "The style is Post-Impressionism, Pointillism",
        "description": "This is a masterpiece of pointillism, capturing the leisurely activities of Parisians at La Grande Jatte island on the Seine River.",
        "appraisal": "I am excited to offer you one hundred million dollars"
    },
    "Georgia-OKeeffe-Red-Canna-1924": {
        "name": " This is Georgia O'Keeffe's Red Canna",
        "remark": 'This is a very famous painting and I think it will sell for a good chunk of change',
        "artist": "The artist is Georgia O'Keeffe",
        "year": "Nineteen twenty four",
        "style": "The style is American Modernism",
        "description": "This is part of O'Keeffe's famous series of flower paintings, Red Canna is celebrated for its vibrant colors and bold depiction.",
        "appraisal": "I would like to make an offer for forty five millions dollars and fifty cents"
    },
    "Pearl-Earring": {
        "name": "This is Johannes Vermeer's The Girl With a Pearl Earrings",
        "remark": "This is a very famous painting and I think it will sell for whole lot of money",
        "artist": "Johannes Vermeer",
        "year": "sixteen sixty five",
        "style": "The style is Dutch Golden Age",
        "description": " This painting is ofter referred to as the 'Mona Lisa of the North', this painting is renowned for its exquisite detail and emotional depth.",
        "appraisal": "How dows one hundred and twenty million dollars sound?"
    },
    "Mountain-AI-Lake-Training-Photos": {
        "name": " I am not sure what the name is but it seems very serene",
        "remark": "I am not sure that this will sell for very much so I do not think it is worth very much",
        "artist": "I do not recognize the artist",
        "year": "I am not sure when this was made",
        "style": "The style seems to be AI-Generated Art",
        "description": "What a nice serene depiction of a mountain and a lake surrounded by trees, it is kind of crazy what AI is capable of in art creation.",
        "appraisal": "Best I can offer is thirty dollars"
    },
    "René-Magritte-The-Son-of-Man-1964": {
        "name": "This is Rene Magritte's The Son Of Man",
        "remark": "This is a very impressive piece of art and I think it will sell for more than you think",
        "artist": "René Magritte",
        "year": "Nineteen sixty four",
        "style": "The style is Surrealism",
        "description": "The painting is a self-portrait with a twist, The Son of Man skillfully plays with reality and perception, featuring a man's face obscured by a floating apple. I think it is pretty cool.",
        "appraisal": "I know that some of the other painting are worth more but I am only willing to offer twenty seven million dollars"
    },
        "Street-Lights-And-Trees-AI-Art": {
        "name": " I am not sure what the name is but it is pretty",
        "remark": " I am not sure that this will sell for very much so I do not think it is worth very much",
        "artist": "I am not familiar with the artist",
        "year": "I honestly do not know",
        "style": "The style seems to be AI-Generated Art",
        "description": "I believe that this is an AI-generated artwork featuring street lights illuminating a path lined with trees, reflecting a quiet urban scene. It is quite pretty but not very authentic.",
        "appraisal":  "All I can offer is fifty dollars"
    }
}

# Class indices mapping
class_indices = {
    0: "Edvard-Munch-The-Scream-1893",
    1: "Georges-Seurat-A-Sunday-Afternoon",
    2: "Georgia-OKeeffe-Red-Canna-1924",
    3: "Pearl-Earring",
    4: "Mountain-AI-Lake-Training-Photos",
    5: "René-Magritte-The-Son-of-Man-1964",
    6: "Street-Lights-And-Trees-AI-Art",    
}

# Function to process image through the model
# Assuming your model expects 224x224 RGB images
def process_image(image):
    # Resize the image to match the input shape required by the model
    processed_image = image.resize((224, 224))

    # Convert the image to a numpy array and normalize pixel values
    processed_image = np.array(processed_image) / 255.0

    # If your model expects a batch of images, add an extra dimension
    processed_image = np.expand_dims(processed_image, axis=0)

    # Predict using the model
    predictions = model.predict(processed_image)
    
    # Get the index of the highest probability prediction
    predicted_index = np.argmax(predictions)    

    # Map the index to the corresponding art name using class_indices
    predicted_art_name = class_indices[predicted_index]
    
    return predicted_art_name

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
    await robot.set_head_angle(degrees(22)).wait_for_completed()
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
        print("Predicted Art Name:", art_name)  # Debugging line
        art_info = get_art_info(art_name)

        await robot.say_text(f"Name of the artwork: {art_info['name']}", duration_scalar=0.65).wait_for_completed()
        await robot.say_text(f"Remark: {art_info['remark']}", duration_scalar=0.65).wait_for_completed()
        await robot.say_text(f"Artist: {art_info['artist']}", duration_scalar=0.65).wait_for_completed()
        await robot.say_text(f"Year: {art_info['year']}", duration_scalar=0.65).wait_for_completed()
        await robot.say_text(f"Style: {art_info['style']}", duration_scalar=0.65).wait_for_completed()
        await robot.say_text(f"Description: {art_info['description']}", duration_scalar=0.65).wait_for_completed()
        await robot.say_text(f"Appraisal Value: {art_info['appraisal']}", duration_scalar=0.65).wait_for_completed()

        await handle_user_response(robot, yes_cube, no_cube)

    # Add event handler for camera cube tap
    camera_cube.add_event_handler(cozmo.objects.EvtObjectTapped, on_camera_cube_tap)

    while True:
        time.sleep(1)

# Run the program
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
