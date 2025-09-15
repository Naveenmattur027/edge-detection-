import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io

def apply_edge_detection(image, blur_kernel_size, blur_sigma, canny_low, canny_high):
    """Apply edge detection to an image with customizable parameters."""
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), blur_sigma)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    return gray, blurred, edges

def main():
    st.set_page_config(
        page_title="Mobile Edge Detection",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    st.title("Snap & Detect Edges in Real Time!")
    st.markdown("Capture photos from your mobile camera and apply edge detection!")
    
    # Initialize session state
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Edge Detection Parameters")
        
        # Gaussian Blur parameters
        st.subheader("Gaussian Blur")
        blur_kernel_size = st.selectbox(
            "Kernel Size",
            options=[3, 5, 7, 9, 11],
            index=1,
            help="Size of the Gaussian kernel (must be odd)"
        )
        blur_sigma = st.slider(
            "Sigma",
            min_value=0.1,
            max_value=3.0,
            value=1.4,
            step=0.1,
            help="Standard deviation for Gaussian kernel"
        )
        
        # Canny Edge Detection parameters
        st.subheader("Canny Edge Detection")
        canny_low = st.slider(
            "Low Threshold",
            min_value=0,
            max_value=255,
            value=50,
            help="Lower threshold for edge detection"
        )
        canny_high = st.slider(
            "High Threshold",
            min_value=0,
            max_value=255,
            value=150,
            help="Upper threshold for edge detection"
        )
    
    # Main content area
    tab1, tab2 = st.tabs([" Mobile Camera", " Upload Image"])
    
    with tab1:
        st.subheader(" Mobile Camera Capture")
        st.info(" Use your mobile camera to take photos and apply edge detection instantly!")
        
        # Mobile camera input - works on all devices including mobile
        camera_image = st.camera_input(
            "Take a picture with your camera",
            help="This works on mobile devices and desktop cameras"
        )
        
        if camera_image is not None:
            # Convert the captured image
            image = Image.open(camera_image)
            
            # Apply edge detection
            gray, blurred, edges = apply_edge_detection(
                image, blur_kernel_size, blur_sigma, canny_low, canny_high
            )
            
            # Store processed image in session state
            st.session_state.processed_image = edges
            
            # Display results
            st.subheader("📸 Results")
            
            # Use columns for side-by-side display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Original Image**")
                st.image(image, use_column_width=True)
            
            with col2:
                st.write("**Edge Detection**")
                st.image(edges, use_column_width=True, channels="GRAY")
            
            # Processing steps in expandable section
            with st.expander("🔍 View Processing Steps", expanded=False):
                step_col1, step_col2 = st.columns(2)
                
                with step_col1:
                    st.write("**Grayscale**")
                    st.image(gray, use_column_width=True, channels="GRAY")
                
                with step_col2:
                    st.write("**Gaussian Blur**")
                    st.image(blurred, use_column_width=True, channels="GRAY")
            
            # Download section
            st.subheader(" Download Results")
            
            # Create download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download original
                buf_orig = io.BytesIO()
                image.save(buf_orig, format='PNG')
                st.download_button(
                    label=" Download Original",
                    data=buf_orig.getvalue(),
                    file_name="mobile_original.png",
                    mime="image/png"
                )
            
            with col2:
                # Download edge detection result
                edges_pil = Image.fromarray(edges)
                buf_edges = io.BytesIO()
                edges_pil.save(buf_edges, format='PNG')
                st.download_button(
                    label=" Download Edges",
                    data=buf_edges.getvalue(),
                    file_name="mobile_edge_detection.png",
                    mime="image/png"
                )
    
    with tab2:
        st.subheader("Upload and Process Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image from your device...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload any image to apply edge detection"
        )
        
        if uploaded_file is not None:
            # Load and display the uploaded image
            image = Image.open(uploaded_file)
            
            # Apply edge detection
            gray, blurred, edges = apply_edge_detection(
                image, blur_kernel_size, blur_sigma, canny_low, canny_high
            )
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Original Image**")
                st.image(image, use_column_width=True)
            
            with col2:
                st.write("**Edge Detection Result**")
                st.image(edges, use_column_width=True, channels="GRAY")
            
            # Additional processing steps
            with st.expander("🔍 View Processing Steps"):
                step_col1, step_col2 = st.columns(2)
                
                with step_col1:
                    st.write("**Grayscale**")
                    st.image(gray, use_column_width=True, channels="GRAY")
                
                with step_col2:
                    st.write("**Gaussian Blur**")
                    st.image(blurred, use_column_width=True, channels="GRAY")
            
            # Download processed image
            edges_pil = Image.fromarray(edges)
            buf = io.BytesIO()
            edges_pil.save(buf, format='PNG')
            
            st.download_button(
                label="Download Edge Detection Result",
                data=buf.getvalue(),
                file_name="uploaded_edge_detection.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
