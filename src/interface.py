import streamlit as st
import cv2
import numpy as np
from main import BasketballShotAnalyzer
import tempfile
import os
from datetime import datetime

class BasketballAnalyzerInterface:
    def __init__(self):
        self.analyzer = None
        self.temp_dir = tempfile.mkdtemp()
        
    def setup_page(self):
        """Setup the Streamlit page configuration"""
        st.set_page_config(
            page_title="Basketball Shot Analyzer",
            page_icon="🏀",
            layout="wide"
        )
        
        st.title("🏀 Basketball Shot Analyzer")
        st.markdown("""
        Analyze your basketball shooting form in real-time using AI.
        Upload a video or use your webcam to get started.
        """)
    
    def create_sidebar(self):
        """Create the sidebar with controls"""
        with st.sidebar:
            st.header("Settings")
            
            # Input source selection
            input_source = st.radio(
                "Select Input Source",
                ["Webcam", "Video File"],
                index=0
            )
            
            video_path = None
            if input_source == "Video File":
                uploaded_file = st.file_uploader(
                    "Upload Video",
                    type=['mp4', 'avi', 'mov'],
                    help="Upload a video file to analyze"
                )
                
                if uploaded_file:
                    # Save uploaded file temporarily
                    video_path = os.path.join(self.temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
            
            # Analysis settings
            st.subheader("Analysis Settings")
            show_trajectory = st.checkbox("Show Ball Trajectory", value=True)
            show_prediction = st.checkbox("Show Trajectory Prediction", value=True)
            show_metrics = st.checkbox("Show Metrics", value=True)
            
            # Recording settings
            st.subheader("Recording")
            record_session = st.checkbox("Record Session", value=False)
            
            settings = {
                'show_trajectory': show_trajectory,
                'show_prediction': show_prediction,
                'show_metrics': show_metrics,
                'record_session': record_session
            }
            
            return video_path, settings
    
    def create_main_content(self):
        """Create the main content area"""
        # Create two columns for the interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Analysis")
            frame_placeholder = st.empty()
            
        with col2:
            st.subheader("Metrics")
            metrics_placeholder = st.empty()
            
            st.subheader("Shot History")
            history_placeholder = st.empty()
        
        return frame_placeholder, metrics_placeholder, history_placeholder
    
    def run(self):
        """Run the interface"""
        self.setup_page()
        
        # Create sidebar and get settings
        video_path, settings = self.create_sidebar()
        
        # Create main content
        frame_placeholder, metrics_placeholder, history_placeholder = self.create_main_content()
        
        # Initialize analyzer
        if video_path:
            self.analyzer = BasketballShotAnalyzer(source=video_path)
        else:
            self.analyzer = BasketballShotAnalyzer(source=0)
        
        # Initialize session state
        if 'shot_history' not in st.session_state:
            st.session_state.shot_history = []
        
        # Main processing loop
        try:
            while True:
                # Process frame
                processed_frame = self.analyzer.process_frame()
                
                if processed_frame is None:
                    break
                
                # Update frame display
                frame_placeholder.image(processed_frame, channels="BGR")
                
                # Update metrics
                if settings['show_metrics']:
                    metrics = self.analyzer.get_current_metrics()
                    metrics_placeholder.write(metrics)
                
                # Check for shot completion
                if self.analyzer.is_shot_complete():
                    shot_data = self.analyzer.get_shot_data()
                    st.session_state.shot_history.append({
                        'timestamp': datetime.now(),
                        'data': shot_data
                    })
                    
                    # Update history
                    history_placeholder.write(st.session_state.shot_history)
                
                # Break if video is finished
                if video_path and not self.analyzer.cap.isOpened():
                    break
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            self.analyzer.cleanup()
    
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)

def main():
    interface = BasketballAnalyzerInterface()
    interface.run()
    interface.cleanup()

if __name__ == "__main__":
    main() 