import streamlit as st 
import pickle as pickle 
import pandas as pd 

## python -m streamlit run streamlit_video.py

def main():
    st.set_page_config(
        page_title = "Boxing Hit Detection Predictor",
        page_icon = ":female_doctor:",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )
    
    with st.container():
        st.title("Boxing Hit Detection Predictor")
        st.write("""
                 This application using a Machine Learning Model to 
                 classify what type of hit you have done! Upload your 
                 boxing video using the container below to 'analyze' your 
                 form.
                 """)
        st.markdown("---")

        #Now let's make some columns 
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write("This is column 1")
        with col2:
            st.write("This is column 2")



if __name__ == "__main__":
    main()