import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import threading

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
        page_title="Webcam Edge Detection",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("🎥 Webcam Edge Detection with OpenCV")
    st.markdown("Capture frames from your webcam and apply edge detection!")
    
    # Initialize session state
    if 'captured_frame' not in st.session_state:
        st.session_state.captured_frame = None
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    # Sidebar for controls
    st.sidebar.header("🔧 Edge Detection Parameters")
    
    # Gaussian Blur parameters
    st.sidebar.subheader("Gaussian Blur")
    blur_kernel_size = st.sidebar.selectbox(
        "Kernel Size",
        options=[3, 5, 7, 9, 11],
        index=1,
        help="Size of the Gaussian kernel (must be odd)"
    )
    blur_sigma = st.sidebar.slider(
        "Sigma",
        min_value=0.1,
        max_value=3.0,
        value=1.4,
        step=0.1,
        help="Standard deviation for Gaussian kernel"
    )
    
    # Canny Edge Detection parameters
    st.sidebar.subheader("Canny Edge Detection")
    canny_low = st.sidebar.slider(
        "Low Threshold",
        min_value=0,
        max_value=255,
        value=50,
        help="Lower threshold for edge detection"
    )
    canny_high = st.sidebar.slider(
        "High Threshold",
        min_value=0,
        max_value=255,
        value=150,
        help="Upper threshold for edge detection"
    )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📷 Webcam Capture", "🖼️ Upload Image", "🎬 Live Stream (Advanced)"])
    
    with tab1:
        st.subheader("Webcam Frame Capture")
        st.info("📌 Click 'Capture Frame' to take a snapshot from your webcam and apply edge detection.")
        
        # Webcam controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📸 Capture Frame", type="primary"):
                try:
                    # Initialize webcam
                    cap = cv2.VideoCapture(0)
                    
                    if not cap.isOpened():
                        st.error("❌ Could not access webcam. Please check your camera permissions.")
                    else:
                        # Capture a single frame
                        ret, frame = cap.read()
                        if ret:
                            # Convert to RGB and store in session state
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.session_state.captured_frame = Image.fromarray(frame_rgb)
                            st.success("✅ Frame captured successfully!")
                        else:
                            st.error("❌ Failed to capture frame from webcam")
                    
                    cap.release()
                    
                except Exception as e:
                    st.error(f"❌ Error accessing webcam: {str(e)}")
                    st.info("💡 Make sure your webcam is not being used by another application.")
        
        with col2:
            if st.button("🗑️ Clear Frame"):
                st.session_state.captured_frame = None
                st.success("Frame cleared!")
        
        # Display captured frame and edge detection result
        if st.session_state.captured_frame is not None:
            # Apply edge detection
            gray, blurred, edges = apply_edge_detection(
                st.session_state.captured_frame, 
                blur_kernel_size, 
                blur_sigma, 
                canny_low, 
                canny_high
            )
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Captured Frame")
                st.image(st.session_state.captured_frame, use_column_width=True)
            
            with col2:
                st.subheader("🔍 Edge Detection")
                st.image(edges, use_column_width=True)
            
            # Additional processing steps
            with st.expander("🔍 View Processing Steps"):
                step_col1, step_col2 = st.columns(2)
                
                with step_col1:
                    st.subheader("Grayscale")
                    st.image(gray, use_column_width=True)
                
                with step_col2:
                    st.subheader("Gaussian Blur")
                    st.image(blurred, use_column_width=True)
            
            # Download option
            if st.button("💾 Download Edge Result"):
                import io
                edges_pil = Image.fromarray(edges)
                buf = io.BytesIO()
                edges_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Click to Download",
                    data=byte_im,
                    file_name="webcam_edge_detection.png",
                    mime="image/png"
                )
        else:
            st.info("👆 Click 'Capture Frame' to get started!")
    
    with tab2:
        st.subheader("Upload and Process Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to apply edge detection"
        )
        
        if uploaded_file is not None:
            # Load and display the uploaded image
            image = Image.open(uploaded_file)
            
            # Apply edge detection
            gray, blurred, edges = apply_edge_detection(
                image, blur_kernel_size, blur_sigma, canny_low, canny_high
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🔍 Edge Detection Result")
                st.image(edges, use_column_width=True)
            
            # Additional processing steps
            with st.expander("🔍 View Processing Steps"):
                step_col1, step_col2 = st.columns(2)
                
                with step_col1:
                    st.subheader("Grayscale")
                    st.image(gray, use_column_width=True)
                
                with step_col2:
                    st.subheader("Gaussian Blur")
                    st.image(blurred, use_column_width=True)
            
            # Download processed image
            if st.button("💾 Download Result", key="upload_download"):
                import io
                edges_pil = Image.fromarray(edges)
                buf = io.BytesIO()
                edges_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Click to Download",
                    data=byte_im,
                    file_name="uploaded_edge_detection.png",
                    mime="image/png",
                    key="upload_download_btn"
                )
    
    with tab3:
        st.subheader("🎬 Live Stream (Experimental)")
        st.warning("⚠️ This feature requires running Streamlit locally and may not work in all browsers.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_stream = st.button("▶️ Start Live Stream", key="start_stream")
        with col2:
            stop_stream = st.button("⏹️ Stop Stream", key="stop_stream")
        
        if start_stream:
            st.session_state.webcam_running = True
        
        if stop_stream:
            st.session_state.webcam_running = False
        
        # Live stream placeholder
        stream_placeholder = st.empty()
        
        if st.session_state.webcam_running:
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not access webcam for live stream")
                    st.session_state.webcam_running = False
                else:
                    fps_counter = 0
                    start_time = time.time()
                    
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read frame")
                            break
                        
                        # Convert and process frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Apply edge detection
                        _, _, edges = apply_edge_detection(
                            pil_image, blur_kernel_size, blur_sigma, canny_low, canny_high
                        )
                        
                        # Display in columns
                        with stream_placeholder.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(frame_rgb, caption="🎥 Live Feed", use_column_width=True)
                            with col2:
                                st.image(edges, caption="🔍 Live Edge Detection", use_column_width=True)
                        
                        fps_counter += 1
                        if fps_counter % 30 == 0:  # Update every 30 frames
                            elapsed = time.time() - start_time
                            fps = fps_counter / elapsed
                            st.sidebar.metric("📊 FPS", f"{fps:.1f}")
                        
                        # Small delay to prevent overwhelming the browser
                        time.sleep(0.1)
                
                cap.release()
                
            except Exception as e:
                st.error(f"❌ Error in live stream: {str(e)}")
                st.session_state.webcam_running = False
        
        if not st.session_state.webcam_running and not start_stream:
            st.info("👆 Click 'Start Live Stream' to begin real-time edge detection")
    
    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        **🎯 How to Use:**
        
        1. **Webcam Capture**: Take snapshots from your webcam and apply edge detection
        2. **Upload Image**: Process any image file with edge detection
        3. **Live Stream**: Real-time processing (experimental, works best locally)
        
        **🔧 Edge Detection Process:**
        
        1. **Grayscale Conversion**: Simplify image processing
        2. **Gaussian Blur**: Reduce noise and smooth edges
        3. **Canny Edge Detection**: Detect edges using gradient analysis
        
        **⚙️ Parameter Tips:**
        - **Lower thresholds**: Detect more edges (including noise)
        - **Higher thresholds**: Detect only strong edges
        - **Larger blur kernel**: More smoothing, fewer noise edges
        - **Higher sigma**: Stronger blur effect
        
        **🔧 Troubleshooting:**
        - Make sure no other applications are using your webcam
        - Try refreshing the page if webcam doesn't work
        - Use 'Upload Image' tab as an alternative to webcam issues
        """)

if __name__ == "__main__":
    main()