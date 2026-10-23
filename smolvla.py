"""Minimal SmolVLA smoke test (run in the `lerobot` conda env)."""
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors  # path may differ by LeRobot version


class smolVLA():

    def __init__(self):
        super().__init__('smolVLA')
        self.MODEL_ID = "lerobot/smolvla_base"   #smolVLA model we want to load
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # searching for a nvidia GPU for computation, if fails, uses CPU (alot slower)
        self.policy = SmolVLAPolicy.from_pretrained(self.MODEL_ID).to(self.device).eval() #Loads / Builds the model, moving its weights into the device (GPU or CPU)
        self.policy.reset()  #Clears the policies action queue for anything remaining.

        # Pre : Takes raw observation dict, normalises using the mean and std deviation from training data, tokenizes language instruction to numbers, moves everything onto the device
        # Post : Takes model output, converts back into robots real units by undoing the normalisation it was trained with.

        self.pre, self.post = make_pre_post_processors(       #Makes pre and post processor pipelines. 
            self.policy.config, self.MODEL_ID,                #config: Which features the model expects (image keys, state size). Model ID: Which processor settings to load
            preprocessor_overrides={"device_processor": {"device": self.device}},    #Forces preprocessor to move tensors to the device specified.
        )
        print("Expected inputs:", self.policy.config.input_features)

        return    
    
    def get_obs(self,frame,joints,task):
        self.obs = {     # the observation dict smolVLA expects (Standardised naming convention)
                "observation.images.camera1": self.to_tensor(frame),
                "observation.images.camera2": self.to_tensor(frame),
                "observation.images.camera3": self.to_tensor(frame),
                "observation.state": joints,   # joint state in the policy's units
                "task": task,               # required: SmolVLA is language-conditioned
            }
        return

    def to_tensor(rgb_uint8):
        '''
        Needed to convert the RGB image into CHW for SmolVLA inference.
        '''
        """HxWx3 uint8 RGB -> 1x3xHxW float in [0,1]. (No BGR swap: Isaac Sim gives RGB;
        if you load files with cv2.imread, convert with cv2.cvtColor(..., COLOR_BGR2RGB) first.)"""
        return torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


    # Placeholder frames; replace with real images (any size, the policy resizes internally)
    #frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    def main(self):
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(self.obs)))
    
        print("Action for this step:", action.shape)   # (1, action_dim): ONE action, not the whole chunk
        print(action)
        return
    

    