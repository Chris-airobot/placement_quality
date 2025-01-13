# import carb
from isaacsim import SimulationApp

CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)


import omni
import numpy as np
from omni.isaac.core import SimulationContext, World
from omni.isaac.core.utils import stage, extensions
from camera_initialization import MyCamera
# from graph_initialization import joint_graph_generation, gripper_graph_generation
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()




class StartSimulation:

    def __init__(self):
        self.usd_path  = "/home/chris/Chris/placement_ws/src/grasp_placement/panda.usd"
        self.prim_path = "/World/franka_alt_fingers"
        self.world = None
        self.simulation_context = None
        self.camera = None

    def start(self):
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane() 

        self.simulation_context = SimulationContext(stage_units_in_meters=1.0)
        # Loading the robot
        stage.add_reference_to_stage(self.usd_path, self.prim_path)
        # Initialize physics
        self.simulation_context.initialize_physics()
        self.simulation_context.play()
        # Loading the action graph

        # Loading the camera
        self.camera = MyCamera()
        self.camera.initialize()

        simulation_app.update()
        self.camera.initialize()

        # Starting the camera
        self.camera.start_camera()

        


        # self.graph_init()


        

def main():

    env = StartSimulation()
    env.start()

    
    while simulation_app.is_running():
        env.simulation_context.step(render=True)

    env.simulation_context.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()
















