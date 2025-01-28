import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from datetime import datetime

def publish_camera_info(camera: Camera, freq):
    from omni.isaac.ros2_bridge import read_camera_info
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_camera_info"
    queue_size = 1
    node_namespace = f""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    writer = rep.writers.get("ROS2PublishCameraInfo")
    camera_info = read_camera_info(render_product_path=render_product)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info["width"],
        height=camera_info["height"],
        projectionType=camera_info["projectionType"],
        k=camera_info["k"].reshape([1, 9]),
        r=camera_info["r"].reshape([1, 9]),
        p=camera_info["p"].reshape([1, 12]),
        physicalDistortionModel=camera_info["physicalDistortionModel"],
        physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
    )
    writer.attach([render_product])
    omni.syntheticdata
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_pointcloud_from_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_pointcloud" # Set topic name to the camera's name
    queue_size = 1
    node_namespace = f""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    # Note, this pointcloud publisher will simply convert the Depth image to a pointcloud using the Camera intrinsics.
    # This pointcloud generation method does not support semantic labelled objects.
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        omni.syntheticdata._syntheticdata.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return

def publish_rgb(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(omni.syntheticdata._syntheticdata.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_depth(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_depth"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                            omni.syntheticdata._syntheticdata.SensorType.DistanceToImagePlane.name
                        )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_camera_tf(camera: Camera):
    camera_prim = camera.prim_path

    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    try:
        # Generate the camera_frame_id. OmniActionGraph will use the last part of
        # the full camera prim path as the frame name, so we will extract it here
        # and use it for the pointcloud frame_id.
        camera_frame_id=camera_prim.split("/")[-1]

        # Generate an action graph associated with camera TF publishing.
        ros_camera_graph_path = "/Graphs/CameraTFActionGraph"

        # If a camera graph is not found, create a new one.
        if not is_prim_path_valid(ros_camera_graph_path):
            (ros_camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": ros_camera_graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("IsaacClock", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("RosPublisher", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                        ("IsaacClock.outputs:simulationTime", "RosPublisher.inputs:timeStamp"),
                    ]
                }
            )

        # Generate 2 nodes associated with each camera: TF from world to ROS camera convention, and world frame.
        og.Controller.edit(
            ros_camera_graph_path,
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("PublishTF_"+camera_frame_id, "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                    ("PublishRawTF_"+camera_frame_id+"_world", "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF_"+camera_frame_id+".inputs:topicName", "/tf"),
                    # Note if topic_name is changed to something else besides "/tf",
                    # it will not be captured by the ROS tf broadcaster.
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:topicName", "/tf"),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:parentFrameId", camera_frame_id),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:childFrameId", camera_frame_id+"_world"),
                    # Static transform from ROS camera convention to world (+Z up, +X forward) convention:
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
                ],
                og.Controller.Keys.CONNECT: [
                    (ros_camera_graph_path+"/OnTick.outputs:tick",
                        "PublishTF_"+camera_frame_id+".inputs:execIn"),
                    (ros_camera_graph_path+"/OnTick.outputs:tick",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:execIn"),
                    (ros_camera_graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishTF_"+camera_frame_id+".inputs:timeStamp"),
                    (ros_camera_graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:timeStamp"),
                ],
            },
        )
    except Exception as e:
        print(e)

    # Add target prims for the USD pose. All other frames are static.
    set_target_prims(
        primPath=ros_camera_graph_path+"/PublishTF_"+camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )
    return



def camera_graph_generation(
    camera: Camera,
    graph_path: str = "/Graphs/ROS_Camera",
    node_namespace: str = "",
    rgb_topic: str = "/rgb",
    depth_topic: str = "/depth",
    depth_pcl_topic: str = "/depth_pcl",
):
    """
    Creates an OmniGraph that publishes:
    1) Camera Info
    2) RGB Image
    3) Depth Image
    4) Depth Pointcloud
    from the specified camera prim via ROS2.

    No conditionals—always publishes the three streams plus camera info.
    """
    camera_prim = camera.prim_path
    frame_id = camera_prim.split("/")[-1]

    # # Stop the timeline so we can safely build the graph
    # timeline = omni.timeline.get_timeline_interface()
    # timeline.stop()

    # Create a new graph with OnPlaybackTick, RunOnce, RenderProduct, ROS2Context
    # Always use "execution" evaluator
    graph_edit_result = og.Controller.edit(
        {
            "graph_path": graph_path, 
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            # Create nodes
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RunOnce", "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame"),
                ("RenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("CameraInfo", "omni.isaac.ros2_bridge.ROS2CameraInfoHelper"),
                ("RGB", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("Depth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("DepthPCL", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            # Set attribute values
            og.Controller.Keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", camera_prim),
                ("CameraInfo.inputs:topicName", "camera_info"),
                ("CameraInfo.inputs:frameId", frame_id),
                ("CameraInfo.inputs:nodeNamespace", node_namespace),
                ("CameraInfo.inputs:resetSimulationTimeOnStop", True),

                ("RGB.inputs:topicName", rgb_topic),
                ("RGB.inputs:type", "rgb"),
                ("RGB.inputs:frameId", frame_id),
                ("RGB.inputs:nodeNamespace", node_namespace),
                ("RGB.inputs:resetSimulationTimeOnStop", True),

                ("Depth.inputs:topicName", depth_topic),
                ("Depth.inputs:type", "depth"),
                ("Depth.inputs:frameId", frame_id),
                ("Depth.inputs:nodeNamespace", node_namespace),
                ("Depth.inputs:resetSimulationTimeOnStop", True),

                ("DepthPCL.inputs:topicName", depth_pcl_topic),
                ("DepthPCL.inputs:type", "depth_pcl"),
                ("DepthPCL.inputs:frameId", frame_id),
                ("DepthPCL.inputs:nodeNamespace", node_namespace),
                ("DepthPCL.inputs:resetSimulationTimeOnStop", True),
            ],
            # Connect execution flows and data
            og.Controller.Keys.CONNECT: [
                # Tick -> RunOnce
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),

                # RunOnce -> RenderProduct
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),

                # RenderProduct -> CameraInfo
                ("RenderProduct.outputs:execOut", "CameraInfo.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "CameraInfo.inputs:renderProductPath"),

                # RenderProduct -> RGB
                ("RenderProduct.outputs:execOut", "RGB.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RGB.inputs:renderProductPath"),

                # RenderProduct -> Depth
                ("RenderProduct.outputs:execOut", "Depth.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "Depth.inputs:renderProductPath"),

                # RenderProduct -> DepthPCL
                ("RenderProduct.outputs:execOut", "DepthPCL.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "DepthPCL.inputs:renderProductPath"),

                # Context -> CameraInfo
                ("Context.outputs:context", "CameraInfo.inputs:context"),
                # Context -> RGB
                ("Context.outputs:context", "RGB.inputs:context"),
                # Context -> Depth
                ("Context.outputs:context", "Depth.inputs:context"),
                # Context -> DepthPCL
                ("Context.outputs:context", "DepthPCL.inputs:context"),
            ],
        },
    )









def start_camera(camera: Camera):
    approx_freq = 30
    publish_camera_tf(camera)
    camera_graph_generation(camera)
    # publish_camera_info(camera, approx_freq)
    # publish_rgb(camera, approx_freq)
    # publish_depth(camera, approx_freq)
    # publish_pointcloud_from_depth(camera, approx_freq)
