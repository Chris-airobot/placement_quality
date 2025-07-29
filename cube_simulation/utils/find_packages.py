"""
All you need to do is simply import the package you want to find for isaac sim, and put it into the "modules"
"""


# import carb
from isaacsim import SimulationApp
import sys

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/home/chris/Chris/placement_ws/src/grasp_placement/panda.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True}

# Example ROS2 bridge sample demonstrating the manual loading of stages and manual publishing of images
simulation_app = SimulationApp(CONFIG)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
import sys
import carb
from pxr import Gf
from pxr import Sdf, UsdLux, Tf

modules = [
    "pxr",
    "isaacsim.core.api",
    "isaacsim.core.utils",
    "warp"
]

for module_name in modules:
    if module_name in sys.modules:
        print(f"\nModule '{module_name}' is loaded.")
        mod = sys.modules[module_name]
        # Attempt to print the __file__ attribute if it exists
        try:
            print(f"  __file__ = {mod.__path__}")
        except AttributeError:
            print("  No __file__ attribute (likely a compiled/binary extension).")
    else:
        print(f"\nModule '{module_name}' is NOT loaded.")



