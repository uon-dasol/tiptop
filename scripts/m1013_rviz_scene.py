"""ROS-side visualization worker, run inside ros2-dev by rviz_m1013_tamp.py.

Only publishes markers and reads TF. Attachment is the task planner's symbolic
state, not a physics constraint. No controller or hardware API is used here.
"""
import json
import queue
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def matrix(p, q):
    x, y, z, w = q
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = p
    return t


def quaternion(t):
    # Stable matrix→quaternion conversion, including rotations near 180 degrees.
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(t[:3, :3]).as_quat()


def main():
    scene = json.loads(sys.stdin.readline())
    p = scene["object_pose"]
    world_obj = matrix(p[:3], [p[4], p[5], p[6], p[3]])
    target = scene["target_xyz"]
    rclpy.init(args=["--ros-args", "-r", "/tf:=/uon_model/tf", "-r", "/tf_static:=/uon_model/tf_static"])
    node = rclpy.create_node("m1013_tamp_object_preview")
    pub = node.create_publisher(MarkerArray, "/uon_model/planning_preview", 10)
    buffer = Buffer(cache_time=Duration(seconds=5))
    listener = TransformListener(buffer, node)
    inbox = queue.Queue()

    def read():
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put({"event": "quit"})

    threading.Thread(target=read, daemon=True).start()

    def tcp_pose():
        tf = buffer.lookup_transform("world", "gripper_tcp", rclpy.time.Time()).transform
        return matrix([tf.translation.x, tf.translation.y, tf.translation.z],
                      [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])

    def marker(index, kind, transform, scale, color):
        m = Marker()
        m.header.frame_id = "world"
        m.ns, m.id, m.type = "m1013_tamp_scene", index, kind
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, transform[:3, 3])
        q = quaternion(transform)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, q)
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    attached, relative, phase = False, None, "cuTAMP: ready"
    start = time.monotonic()
    ready = False
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=.04)
        try:
            tcp = tcp_pose()
        except Exception:
            if time.monotonic()-start > 12:
                raise RuntimeError("Missing world→gripper_tcp TF; no motion should start")
            continue
        if attached:
            world_obj = tcp @ relative
        while not inbox.empty():
            command = inbox.get()
            event = command["event"]
            if event == "attach":
                relative = np.linalg.inv(tcp) @ world_obj
                attached, phase = True, "cuTAMP: Pick / attached"
            elif event == "detach":
                attached, phase = False, "cuTAMP: Place / released"
            elif event == "quit":
                print(json.dumps({"event": "quit", "world_from_object": world_obj.tolist()}), flush=True)
                node.destroy_node()
                rclpy.shutdown()
                return
            print(json.dumps({"event": event, "world_from_object": world_obj.tolist()}), flush=True)
        obj = marker(0, Marker.MESH_RESOURCE, world_obj, [1., 1., 1.], [.72, .55, .29, 1.])
        obj.mesh_resource = "file:///workspaces/ros2_ws/build/tiptop_grasp_preview/pepper.stl"
        goal = np.eye(4)
        goal[:3, 3] = target
        surface = marker(1, Marker.CUBE, goal, [.18, .18, .02], [.15, .8, .35, .5])
        label = world_obj.copy()
        label[:3, 3] += [0, 0, .12]
        text = marker(2, Marker.TEXT_VIEW_FACING, label, [0., 0., .018], [1., .85, .35, 1.])
        text.text = phase
        msg = MarkerArray(markers=[obj, surface, text])
        pub.publish(msg)
        if not ready and pub.get_subscription_count() > 0:
            ready = True
            print(json.dumps({"event": "ready", "world_from_object": world_obj.tolist()}), flush=True)
        elif not ready and time.monotonic()-start > 12:
            raise RuntimeError("RViz must subscribe to /uon_model/planning_preview before execution")


if __name__ == "__main__":
    main()
