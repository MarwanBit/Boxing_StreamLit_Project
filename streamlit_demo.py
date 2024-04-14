import streamlit as st 
import pickle as pickle 
import pandas as pd 

#Now let's try loading up our other stuff
import video_loading as vl


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

if __name__ == "__main__":
    main_loop()
