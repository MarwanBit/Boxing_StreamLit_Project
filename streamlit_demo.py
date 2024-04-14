import streamlit as st 
import pickle as pickle 
import pandas as pd 

#Now let's try loading up our other stuff
import video_loading as vl
import torch

model = torch.load("movenet_lightning.pth")


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
    st.write(str(model))

if __name__ == "__main__":
    main_loop()
