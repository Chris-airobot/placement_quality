import omni.graph.core as og
from omni.isaac.ros2_bridge.scripts.og_shortcuts.og_utils import Ros2JointStatesGraph

def joint_graph_generation():
    keys = og.Controller.Keys

    robot_frame_id= "World/franka_alt_fingers"
    ros_joint_state_path = "/Graphs/ROS_JointStates"
    (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
        {
            "graph_path": ros_joint_state_path, 
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacClock", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("IsaacArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                ("RosContext", "omni.isaac.ros2_bridge.ROS2Context"),
                ("RosSubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("RosPublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
            ],

            keys.SET_VALUES: [
                ("RosPublishJointState.inputs:topicName", "/isaac_joint_states"),
                ("RosPublishJointState.inputs:targetPrim", robot_frame_id),
                ("RosPublishJointState.inputs:queueSize", 10),

                ("RosSubscribeJointState.inputs:topicName", "/isaac_joint_commands"),
                ("RosPublishJointState.inputs:queueSize", 10),
 
                ("IsaacArticulationController.inputs:targetPrim", robot_frame_id),
            ],

            keys.CONNECT: [
                ("OnTick.outputs:tick", "RosSubscribeJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "RosPublishJointState.inputs:execIn"),

                ("IsaacClock.outputs:simulationTime", "RosPublishJointState.inputs:timeStamp"),

                ("RosContext.outputs:context", "RosPublishJointState.inputs:context"),
                ("RosContext.outputs:context", "RosSubscribeJointState.inputs:context"),

                ("RosSubscribeJointState.outputs:execOut", "IsaacArticulationController.inputs:execIn"),
                ("RosSubscribeJointState.outputs:effortCommand", "IsaacArticulationController.inputs:effortCommand"),
                ("RosSubscribeJointState.outputs:jointNames", "IsaacArticulationController.inputs:jointNames"),
                ("RosSubscribeJointState.outputs:positionCommand", "IsaacArticulationController.inputs:positionCommand"),
                ("RosSubscribeJointState.outputs:velocityCommand", "IsaacArticulationController.inputs:velocityCommand"),



            ]
        }
    )


def gripper_graph_generation():
    keys = og.Controller.Keys

    articulation_root_prim= "World/franka_alt_fingers"
    gripper_prim= "World/franka_alt_fingers/panda_hand"

    ros_gripper_path = "/Graphs/Gripper_Controller"
    (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
        {
            "graph_path": ros_gripper_path, 
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("RosContext", "omni.isaac.ros2_bridge.ROS2Context"),
                ("RosSubscriberOpen", "omni.isaac.ros2_bridge.ROS2Subscriber"),
                ("RosSubscriberClose", "omni.isaac.ros2_bridge.ROS2Subscriber"),
                ("IsaacGipperController", "omni.isaac.manipulators.IsaacGripperController"),

                ("GripperSpeedArray", "omni.graph.nodes.ConstructArray"),
                ("ClosePositionArray", "omni.graph.nodes.ConstructArray"),
                ("OpenPositionArray", "omni.graph.nodes.ConstructArray"),
                ("ArrayJointNames", "omni.graph.nodes.ConstructArray"),
                ("ArrayJointNames_add", "omni.graph.nodes.ArrayInsertValue"),

                ("Speed", "omni.graph.nodes.ConstantDouble"),
                ("CloseJointLimit", "omni.graph.nodes.ConstantDouble"),
                ("OpenJointLimit", "omni.graph.nodes.ConstantDouble"),


            ],

            keys.SET_VALUES: [
                ("Speed.inputs:value", 0.005),
                ("CloseJointLimit.inputs:value", 0.0),
                ("OpenJointLimit.inputs:value", 0.04),

       
                ("ArrayJointNames.inputs:arrayType", "token[]"),
                ("ArrayJointNames.inputs:input0", "panda_finger_joint1"),

                ("ArrayJointNames_add.inputs:index", 1),
                ("ArrayJointNames_add.inputs:value", "panda_finger_joint2"),


       
 
                ("RosSubscriberOpen.inputs:messageName", "Empty"),
                ("RosSubscriberOpen.inputs:messagePackage", "std_msgs"),
                ("RosSubscriberOpen.inputs:messageSubfolder", "msg"),
                ("RosSubscriberOpen.inputs:topicName", "open_gripper"),
                ("RosSubscriberOpen.inputs:queueSize", 10),

                ("RosSubscriberClose.inputs:messageName", "Empty"),
                ("RosSubscriberClose.inputs:messagePackage", "std_msgs"),
                ("RosSubscriberClose.inputs:messageSubfolder", "msg"),
                ("RosSubscriberClose.inputs:topicName", "close_gripper"),
                ("RosSubscriberClose.inputs:queueSize", 10),

                ("IsaacGipperController.inputs:articulationRootPrim", articulation_root_prim),
                ("IsaacGipperController.inputs:gripperPrim", gripper_prim),

                ("GripperSpeedArray.inputs:arraySize", 1),
                ("GripperSpeedArray.inputs:arrayType", "auto"),

                ("ClosePositionArray.inputs:arraySize", 1),
                ("ClosePositionArray.inputs:arrayType", "auto"),

                ("OpenPositionArray.inputs:arraySize", 1),
                ("OpenPositionArray.inputs:arrayType", "auto"),

                

            ],

            keys.CONNECT: [
                ("OnTick.outputs:tick", "RosSubscriberOpen.inputs:execIn"),
                ("OnTick.outputs:tick", "RosSubscriberClose.inputs:execIn"),

                ("RosContext.outputs:context", "RosSubscriberOpen.inputs:context"),
                ("RosContext.outputs:context", "RosSubscriberClose.inputs:context"),

                ("RosSubscriberOpen.outputs:execOut", "IsaacGipperController.inputs:open"),
                ("RosSubscriberClose.outputs:execOut", "IsaacGipperController.inputs:close"),

                ("Speed.inputs:value", "GripperSpeedArray.inputs:input0"),
                ("CloseJointLimit.inputs:value", "ClosePositionArray.inputs:input0"),
                ("OpenJointLimit.inputs:value", "OpenPositionArray.inputs:input0"),

                


                ("GripperSpeedArray.outputs:array", "IsaacGipperController.inputs:gripperSpeed"),
                ("ClosePositionArray.outputs:array", "IsaacGipperController.inputs:closePosition"),
                ("OpenPositionArray.outputs:array", "IsaacGipperController.inputs:openPosition"),
                ("ArrayJointNames.outputs:array", "ArrayJointNames_add.inputs:array"),
                ("ArrayJointNames_add.outputs:array", "IsaacGipperController.inputs:jointNames"),
            ]
        }
    )
