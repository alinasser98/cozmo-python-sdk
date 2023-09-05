import cozmo
from cozmo.util import degrees, distance_mm

UNIT_DISTANCE = 100  # Assuming 1 unit is 100mm

def getHawkID():
    myID = 'nssr'  # Replace 'YOURHAWKID' with your actual Hawk ID
    return [myID]

def alternate_movement(robot, start, end):
    x_diff = end[0] - start[0]
    y_diff = end[1] - start[1]

    while x_diff != 0 or y_diff != 0:
        if y_diff != 0:
            # Move vertically
            direction = 1 if y_diff > 0 else -1
            robot.drive_straight(distance_mm(UNIT_DISTANCE * direction), cozmo.util.speed_mmps(150)).wait_for_completed()
            y_diff -= direction
        if x_diff != 0:
            # Turn and move horizontally
            direction = 1 if x_diff > 0 else -1
            robot.turn_in_place(degrees(90 * direction)).wait_for_completed()
            robot.drive_straight(distance_mm(UNIT_DISTANCE), cozmo.util.speed_mmps(150)).wait_for_completed()
            robot.turn_in_place(degrees(-90 * direction)).wait_for_completed()
            x_diff -= direction

    robot.play_anim(name="Var3").wait_for_completed()

def cozmo_program_var3(robot: cozmo.robot.Robot):
    target = (x, y)  # Set x and y to your target values
    alternate_movement(robot, (0, 0), target)

    # Starting from (a, b)
    start = (a, b)  # Set a and b to your starting values
    target = (x, y)  # Set x and y to your target values
    alternate_movement(robot, start, target)

cozmo.run_program(cozmo_program_var3, use_viewer=False, force_viewer_on_top=False)