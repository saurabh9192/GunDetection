import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO  # Adjust this to your model's library

# Load your model (replace 'best.pt' with your trained model file)
MODEL_PATH = "best.pt"  # relative path
model = YOLO(MODEL_PATH)

# App title and description
st.title("Weapon Recognition App")
st.write("Upload an image or video for object detection, or use your webcam.")

# Sidebar for input options
st.sidebar.header("Input Options")
input_type = st.sidebar.radio(
    "Choose an input source:",
    ("Upload Image/Video", "Webcam")
)

# Detection function
def detect_objects(image):
    # Perform inference using the model
    results = model(image)
    # Render results on the original image
    annotated_image = results[0].plot()  # Adjust based on your model's output
    return annotated_image

# Initialize session state for webcam control
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# Process uploaded file
if input_type == "Upload Image/Video":
    uploaded_file = st.sidebar.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            # Read the image file
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Processing...")
            
            # Convert image to OpenCV format
            image_np = np.array(image)
            processed_image = detect_objects(image_np)
            
            # Display the results
            st.image(processed_image, caption="Detected Objects", use_column_width=True)

        elif file_type == 'video':
            st.video(uploaded_file)
            st.write("Video processing is not yet implemented in this demo.")
        else:
            st.error("Unsupported file type!")

elif input_type == "Webcam":
    # Webcam control buttons
    start_webcam = st.button("Start Webcam", key="start_webcam")
    stop_webcam = st.button("Stop Webcam", key="stop_webcam")

    if start_webcam:
        st.session_state.webcam_active = True

    if stop_webcam:
        st.session_state.webcam_active = False

    # Webcam processing
    if st.session_state.webcam_active:
        st.write("Starting webcam...")
        cap = cv2.VideoCapture(0)  # 0 is the default webcam
        stframe = st.empty()  # Placeholder for live feed

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture webcam feed.")
                break
            
            # Process the frame for object detection
            processed_frame = detect_objects(frame)
            
            # Convert the frame (BGR to RGB for Streamlit)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the processed frame
            stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)
        
        # Release the webcam when stopped
        cap.release()
        cv2.destroyAllWindows()


