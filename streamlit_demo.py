import streamlit as st 
import pickle as pickle 
import pandas as pd 

#Now let's try loading up our other stuff
import torch


from movenet import get_pose_net
 
import cv2
import model_factory
import torch.nn.functional as F


model = model_factory.load_model("movenet_lightning")

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}



def main_loop():
    st.set_page_config(
        page_title = "Boxing Hit Detection Predictor",
    )
    st.title("Boxing Hit Detection Predictor")
    st.markdown("---")
    st.write("""
             This application uses Machine Learning Models, particularly recurrent Neural Networks to 
             take a video of your Boxing moves and to classify the type of hit, particularly classifying whether
             the hit was a left jab, right jab, hook, uppercut, etc with given confidence levels. To utilize the demo,
             please upload your video and the model will predict the hit that you've done!
             """)
    st.markdown("---")
    file = st.file_uploader("Video", ["mp4", "MOV", "AVI"])
    if file:
        st.video(file)
        #Now we need to use this to 
        #Now let's try to do prediction
        model = model_factory.load_model("movenet_lightning")
        input_size = 192
        cuda = False
        device = torch.device("cpu")


        video_path = "data/video/id0_jab_1.mp4"
        cap = cv2.VideoCapture(video_path)
        #load this 
        first_frame = cap.read()

        #Let's evaluate the model
        first_frame = first_frame[1]
        model = model.to(device)
        model.eval()
        model.zero_grad()
        first_frame = torch.Tensor(first_frame)
        # print(first_frame)
        # print(first_frame.shape)

        channel_1 = torch.empty((1, 1920, 1080))
        channel_2 = torch.empty((1, 1920, 1080))
        channel_3 = torch.empty((1, 1920, 1080))

        channel_1 = first_frame[:,:,0]
        channel_2 = first_frame[:,:,1]
        channel_3 = first_frame[:,:,2]
        # print(channel_1.shape)
        channel_1 = channel_1[None, None, None, :,:]
        channel_2 = channel_2[None, None, None, :,:]
        channel_3 = channel_3[None, None, None, :,:]
        assert channel_1.shape == (1,1,1, 1920, 1080)
        # print(channel_1.shape)

        channel_1 = F.interpolate(channel_1,size = (1,192,192))
        channel_2 = F.interpolate(channel_2, size = (1,192,192))
        channel_3 = F.interpolate(channel_3, size=(1,192,192))
        # print(channel_1.shape)

        new_output = torch.empty((1,192,192,3))
        new_output[0,:,:,0] = channel_1[0,0,0,:,:]
        new_output[0,:,:,1] = channel_2[0,0,0,:,:]
        new_output[0,:,:,2] = channel_3[0,0,0,:,:]

        # print(new_output.shape) 

        new_output = new_output.to(device)
        output = model(new_output)
        # print(output.shape)
        st.write(output)

    

if __name__ == "__main__":
    main_loop()
