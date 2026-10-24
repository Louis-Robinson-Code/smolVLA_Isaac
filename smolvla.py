"""(run in the `lerobot` conda env)."""
'''
conda activate /home/louis/miniconda3/envs/lerobot
python3 smolvla.py
'''


import numpy as np
import torch
import socket
import threading

from tcp_helper import send_msg, recv_msg



from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors  # path may differ by LeRobot version


'''
Listening for SMolVLA input over TCP, and sending back the action output.
'''

class Streamer():

    def __init__(self,smolvla):
        self.smolvla = smolvla  #shared instance of smolvla to send the joints,image,and
        bind_ip = "127.0.0.1" # binding our client to local host
        port=5301
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # re-use address to allow quick rebind of a port
        self.server.bind((bind_ip,port))
        self.server.listen(5) # listening with a max backlog of 5
        print(f"TCP listening on port {bind_ip} : {port}")

        self.action=None
        return
        

    def handle_client(self, client_socket):
        try:
            while True:
                obs = recv_msg(client_socket)           # obs = dict: images, state, task string
                frame=obs.get("image")
                #print(f"Frame: {frame}")
                joints=obs.get("state")
                task=obs.get("task")
                if frame is not None and joints is not None and task is not None:
                    print(f"Sending frame size: {frame.shape}, joints: {joints}, task: {task} to smolVLA")
                    self.smolvla.get_obs(frame,joints,task)
                    self.smolvla.get_action()
                    self.action=self.smolvla.action
                send_msg(client_socket, self.action)
                    
        except ConnectionError:
            print("Client disconnected")
        finally:
            client_socket.close()
        return

    
    def main(self):
        while True:
            client,addr = self.server.accept()
            print(f"Accepted connection from: {addr[0]}:{addr[1]}")
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_handler=threading.Thread(target=self.handle_client, args=(client,))
            client_handler.start()
        return







class smolVLA():

    def __init__(self):
        self.action=None
        self.obs=None
        
        self.MODEL_ID = "lerobot/smolvla_base"   #smolVLA model we want to load
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # searching for a nvidia GPU for computation, if fails, uses CPU (alot slower)
        self.policy = SmolVLAPolicy.from_pretrained(self.MODEL_ID).to(self.device).eval() #Loads / Builds the model, moving its weights into the device (GPU)
        self.policy.reset()  #Clears the policies action queue for anything remaining.

        # Pre : Takes raw observation dict, normalises using the mean and std deviation from training data, tokenizes language instruction to numbers, moves everything onto the device
        # Post : Takes model output, converts back into robots real units by undoing the normalisation it was trained with.

        self.pre, self.post = make_pre_post_processors(       #Makes pre and post processor pipelines. 
            self.policy.config, self.MODEL_ID,                #config: Which features the model expects (image keys, state size). Model ID: Which processor settings to load
            preprocessor_overrides={"device_processor": {"device": self.device}},    #Forces preprocessor to move tensors to the device specified.
        )

        '''
        Wobbles in smolVLA output could be down to missmatch in safetensors keynames which doesnt normalize input and denormalize output.
        This is a temporary workaround for a known issue in the pre and post processors where the "action" key is missing from the stats dicts.
        https://github.com/huggingface/lerobot/issues/1763#issuecomment-3703499370
        '''

        EMBODIMENT = "so100.buffer.action" 
        for step in list(self.pre.steps) + list(self.post.steps):
            for attr in ("stats", "_tensor_stats"):
                d = getattr(step, attr, None)
                if isinstance(d, dict) and EMBODIMENT in d and "action" not in d:
                    d["action"] = d[EMBODIMENT]
                    print("patched", type(step).__name__, attr)
        
    
    def get_obs(self,frame,joints,task):
        #print("Received frame size:", frame.shape, "joints:", joints, "task:", task) # converting state to tensor for PyTorch model input
        state = torch.tensor(list(joints), dtype=torch.float32).unsqueeze(0)  # (1, 6)
        self.obs = {     # the observation dict smolVLA expects (Standardised naming convention)
                #"observation.images.camera1": self.to_tensor(frame),    #top
                "observation.images.camera2": self.to_tensor(frame),    #wrist
                #"observation.images.camera3": self.to_tensor(frame),    #side
                "observation.state": state,   # joint state in the policy's units
                "task": task,               # required: SmolVLA is language-conditioned
            }
        return
    
    def get_action(self):
        with torch.inference_mode():
            self.action=self.post(self.policy.select_action(self.pre(self.obs)))
        return

    def to_tensor(self,rgb_uint8):
        '''
        Needed to convert the RGB image into CHW for SmolVLA inference.
        '''
        """HxWx3 uint8 RGB -> 1x3xHxW float in [0,1]. """
        return torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


    # Placeholder frames; replace with real images (any size, the policy resizes internally)
    #frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    



if __name__ == "__main__":
    smolvla = smolVLA()
    streamer = Streamer(smolvla)

    #starting the streamer and smolvla in separate threads so they can run in parallel
    streamer_thread = threading.Thread(target=streamer.main)
    streamer_thread.start()
    #smolvla_thread = threading.Thread(target=smolvla.main)
    #smolvla_thread.start()
    
    print("Done")
    

    