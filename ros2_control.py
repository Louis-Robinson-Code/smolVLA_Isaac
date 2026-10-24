import threading
import subprocess
import rclpy
import numpy as np
import socket
from tcp_helper import send_msg, recv_msg


from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Joy
from geometry_msgs.msg import Twist

#from smolvla import smolVLA





#---Setting joint limits----#
upper_lim=[2,3,0,0.95,1.2,3.14159,2]
lower_lim=[-2,0,-3.1415752,-2.15,-3.1415752,-0.199997279]




'''
ROS2 Bridge to Connect Isaac sim S0-100 arm joint state to SmolVLA
1. --Use start_ros2.py to load up isaac sim with the ros2 bridge--
2. -- Run with : 
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 /home/louis/nvidia/FinalYearProject/ros2_control.py


References:
ROS 2 install:  https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
                https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#install-ros-2
                https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#using-terminal
                https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#setting-up-workspaces

Setting up the Action Graph for ROS2 pub, sub, and ticks: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_manipulation.html#ros2-joint-control-extension-python-scripting

Pub/Sub commands to interact with Isaac: https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html

Ros2 Joy : https://index.ros.org/p/joy/
'''

# Need to find a way to convert the image into a format for smolVLA / Lerobot.
# import the policy, dont need to define a port as thats for hardware.

'''
    TO:do
    SET PUBLISHER AND IMAGES TO ONLY UPDATE THE VALUES FOR IMAGE_ARRAY AND JOINT_ARRAY. WE ONLY WANT ONE INSTANCE OF STREAM HANDLING THE SENDING OTHERWISE
    THEY CAN OVERLAP AND CRASH (CURRRENTLY HAPPENING)

'''


class Joy_Sub(Node):
    '''
    Listening for controller inputs and starting/stopping teleop control state
    '''

    def __init__(self):
        super().__init__('controller_listener')

        self.teleop_state=False #flag for teleoperation state
        self.time_at_flip=self.get_clock().now() # time when teleop state changes

        self.current_joints=self.create_subscription(JointState,'joint_states',self.joints_callback,20) #listening for current joint positions (feedback control later?)
        self.sub=self.create_subscription(Joy,'joy', self.joy_callback, 10)
        self.pub=self.create_publisher(JointState,'joint_command',10)
        
        self.controller = Joy()
        self.teleop_control = JointState()
        self.teleop_control.name = ["shoulder_pan", "shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
        self.target_pos=np.array([0.0,0.0,0.0,0.0,-1.5708,0.0])
        self.max_speed=0.5
        self.last_time= self.get_clock().now()

        return

    def joints_callback(self,msg):  
        #   nothing for now
        return

    def start_teleop_control(self):

        '''
        Converting controller inputs to joint commands, and publishing them.
        '''
        #--- fetching time for dt ---#
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        

        #---fetching controller input----#
        input=np.array([self.controller.axes[0],self.controller.axes[1],-self.controller.axes[4],self.controller.axes[2]-self.controller.axes[5],self.controller.buttons[4]-self.controller.buttons[5],self.controller.buttons[1]-self.controller.buttons[2]])
        
        #----Incrementing the target pos in the desired direction ---#
        self.target_pos+= input* self.max_speed * dt #radians/s

        #--- Enforcing joint limits ----#
        for i, radians in enumerate(self.target_pos):
            self.target_pos[i]=np.clip(self.target_pos[i],lower_lim[i],upper_lim[i])
            #print(self.target_pos)

        #---Setting the joint command to the calculated pos ---#
        self.teleop_control.position = self.target_pos.tolist()
        
        return self.pub.publish(self.teleop_control) #publish the joint commands

  
    def joy_callback(self, msg):
        '''
        Responsible for listening for controller inputs and starting teleop.
        '''
        #----Getting the controller inputs----#
        self.controller.header = msg.header
        self.controller.axes = [
            msg.axes[0], #left stick angular (starts -0.0: left is 1 : right is -1)
            msg.axes[1], #left stick linear (starts -0.0: up is 1 : down is -1)
            msg.axes[2], #left trigger (starts 1.0: pressing goes to -1, positions hold value)
            msg.axes[3], #right stick angular (starts -0.0: left is 1 : right is -1)
            msg.axes[4], #right stick linear (starts -0.0: up is 1 : down is -1)
            msg.axes[5], #right trigger (starts 1.0: pressing goes to -1, positions hold value)
            msg.axes[6], #Dpad angular (starts:0, left = -1, right = 1)
            msg.axes[7]] #Dpad linear (starts:0, down = -1, up = 1)
        self.controller.buttons = [
            msg.buttons[0], #a
            msg.buttons[1], #b
            msg.buttons[2], #x
            msg.buttons[3], #y
            msg.buttons[4], #lb
            msg.buttons[5], #rb
            msg.buttons[6], #tab out
            msg.buttons[7], #menu
            msg.buttons[8], #xbox
            msg.buttons[9], #Left stick
            msg.buttons[10], #right stick
            msg.buttons[11]] #middle button thing?

        #---teleop-control---#
        self.time_now=self.get_clock().now() #getting current time

        #if the xbox button has been pressed to start/stop teleop
        if ((self.time_now-self.time_at_flip).nanoseconds * 1e-9 > 0.3) and self.controller.buttons[8] is 1:    #small delay used to ignore multiple input presses due to high Hz
            self.teleop_state = not self.teleop_state
            self.time_at_flip = self.time_now
            print(f"Teleoperation : ", self.teleop_state)
            
        #if xbox button pressed to start teleop, start formulating joint commands from the controller.
        if self.teleop_state is True:
            self.start_teleop_control()

        return


class Wrist_Camera_Subscriber(Node):
    def __init__(self,stream):
        super().__init__('wrist_camera_listener')
        self.stream=stream      #sharing the stream class instance
        self.latest_frame = None
        self.create_subscription(
            Image,  #image instead of jointstate
            'wrist_camera',                         
            self.image_callback,
            qos_profile_sensor_data)    # best-effort, small queue. Publisher still sends frames if it doesnt get acknowledgement. (leaky downstream)

    def image_callback(self, msg: Image):
        self.latest_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
        
        self.stream.image_array=self.latest_frame   #setting the latest frame to send over TCP
        #print("Received Image")


class Publisher(Node):
    def __init__(self, hz=30):
        super().__init__('joint_publisher')
        self.pub = self.create_publisher(JointState,'joint_command',10)
        self.pos = None
        #timer_period=0.5 #seconds
        #self.timer= self.create_timer(timer_period, self.timer_callback)
        
        self.create_timer(1.0 / hz, self.publish_action)
        return


    def publish_action(self):
        msg=JointState()
        msg.name = ["shoulder_pan", "shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
        if self.pos is not None:
            #--- Enforcing joint limits ----#
            for i, radians in enumerate(self.pos):
                self.pos[i]=np.clip(self.pos[i],lower_lim[i],upper_lim[i])

            msg.position = self.pos # in radians
            self.pub.publish(msg)
            print(f"Publishing: {msg.position}")
        return


class Subscriber(Node):

    def __init__(self,stream):
        super().__init__('joint_subscriber')
        self.stream=stream
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        
        self.subscription  
        self.joint_pos=None# prevent unused variable warning

    def listener_callback(self, msg):
        self.joint_pos=msg.position
        #joint_pos_to_degree = [np.rad2deg(x) for x in self.joint_pos]  #converting to degrees for smolVLA
        self.stream.state_array=self.joint_pos  #setting the updated joint pos to TCP streamers input
        #self.get_logger().info('I heard: "%s"' % msg.position)
        return


'''
Streams the ROS2 outputs (image,joint states, and task) across TCP for SmolVLA to listen on

Ros2 and lerobot are incompatible and therefore must be ran under different environements, hence
the need for TCP.
'''
class Streamer(Node):
    def __init__(self,publisher, hz=30, ):

        super().__init__('ROS_TCP')
        self.publisher=publisher
        self.state_array=None
        self.image_array=None
        self.prompt=None

        self.prev_state_array=None
        self.prev_image_array=None

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # TCP_NODELAY sends small packets immediately. (Doesnt group TCP requests for efficiency)
        self.socket.connect(("127.0.0.1", 5301))
        print(f"ROS2 TCP Streamer Initialised")

        self.create_timer(1.0 / hz, self.send_obs)  #ros2 timer to send the latest observation to smolVLA at a fixed frequency (hz)
        #stream_main = threading.Thread(target=self.main, daemon=True)
        #stream_main.start()
    
        
    
    def send_obs(self):
        #print(f"state_array: {self.state_array} ")
        #if self.state_array != self.prev_state_array or self.image_array != self.prev_image_array:
        send_msg(self.socket, {"state": self.state_array, "image": self.image_array, "task": self.prompt})
        action = recv_msg(self.socket)
        if action is not None:  #action is received from smolVLA and is a tensor of joint angles in degrees, which needs to be converted to radians for ROS2 publishing
            tensor_to_degree_action = np.asarray(action, dtype=np.float64).reshape(-1).tolist()  #tensor to list for publishing
            degree_to_radian_action = [np.deg2rad(x) for x in tensor_to_degree_action]  #converting to radians for ROS2 publishing
            self.publisher.pos = degree_to_radian_action    #publishing to joint publisher node to send to isaac sim
            print(f"Publishing: {self.publisher.pos}")
    
        return

    '''
    def main(self):
        try:
            print("main")
            send_obs_timer=threading.Timer(2.0,self.send_obs)
            send_obs_timer.start()
        except KeyboardInterrupt:
            pass
        finally:
            send_obs_timer.cancel()
        return
    '''
    
                

def main(args=None):


    task_prompt = "Pick up the red cube"

    #running the ros2 joy_node to start the controller publisher
    joy_node = subprocess.Popen(['ros2','run','joy', 'joy_node'])

    rclpy.init(args=args)
    publisher = Publisher()
    stream = Streamer(publisher)  #initialising the TCP streamer with the publisher to send joint commands to
    stream.prompt=task_prompt   #setting the TCP streams task prompt

    subscriber = Subscriber(stream)
    
    wrist_image_subscriber = Wrist_Camera_Subscriber(stream)
    teleop = Joy_Sub()
    

    #VLA = smolVLA()

    executor = MultiThreadedExecutor()
    executor.add_node(subscriber)
    executor.add_node(publisher)
    executor.add_node(wrist_image_subscriber)
    executor.add_node(teleop)
    executor.add_node(stream)
    #executor.add_node(VLA)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass
    finally:
        #stream.stop() # Cleaning up the TCP streamer
        joy_node.terminate() # Cleaning up the joy_node process
        subscriber.destroy_node()
        #VLA.destroy_node()
        publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()






# How many keys
# All paper work
# Next service
# Corrosion
# Video
# 
#
#


'''
Original script:

rclpy.init()    #initialises the ROS2 client library
node = rclpy.create_node('smolVLA_controller') #Creates a node for smolVLA
pub = node.create_publisher(JointState, 'joint_command', 10) #creates a publisher for the topic "joint_command". 10 is queue depth.
sub = node.create_subscriber(JointState, 'joint_states', 10)

# Spin in a separate thread
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True) #loop service for nodes internal timers, callback, clock, etc. Seperate thread or it blocks.
thread.start()

joint_state_position = JointState() #Holds both name and position.

#defining the joints name and position
joint_state_position.name = ["shoulder_pan", "shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
joint_state_position.position = [0.0,0.0,0.3,-1.3,0.3,1.0] # in radians

rate = node.create_rate(10) #sends 10 times per sec
try:
    while rclpy.ok():
        pub.publish(joint_state_position)   #publishing the joints

        rate.sleep() # sends and sleeps, waiting for the spin thread to reactivate
except KeyboardInterrupt:
    pass
rclpy.shutdown()
thread.join()
'''