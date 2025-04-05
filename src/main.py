import cv2
import numpy as np
import argparse
from typing import Optional
from ball_tracker import BallTracker
from pose_analyzer import PoseAnalyzer
import time

class BasketballShotAnalyzer:
    def __init__(self, source=0):
        """
        Initialize basketball shot analyzer
        Args:
            source: Camera index or video file path
        """
        print("Initializing Basketball Shot Analyzer...")
        
        # Initialize components
        self.ball_tracker = BallTracker()
        self.pose_analyzer = PoseAnalyzer()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize shot tracking
        self.current_shot = None
        self.shot_history = []
        
        print(f"Video properties: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
    
    def run(self):
        """Run the basketball shot analyzer"""
        print("Starting analysis...")
        
        # Initialize FPS counter
        prev_time = time.time()
        fps_display = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            fps_display = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
            prev_time = current_time
            
            # Process frame
            processed_frame = self.process_frame()
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Basketball Shot Analyzer', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def process_frame(self) -> np.ndarray:
        """
        Process a single frame
        Returns:
            Processed frame with visualizations
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Detect ball
        ball_center, ball_bbox, ball_conf = self.ball_tracker.detect(frame)
        
        # Detect and analyze pose
        landmarks, frame_with_pose = self.pose_analyzer.detect_pose(frame)
        pose_metrics = self.pose_analyzer.analyze_pose(frame, landmarks, ball_center)
        
        # Draw visualizations
        output_frame = self.draw_visualizations(
            frame_with_pose,
            ball_center,
            ball_bbox,
            landmarks,
            pose_metrics
        )
        
        return output_frame
    
    def get_current_metrics(self) -> dict:
        """Get current analysis metrics"""
        return {
            'Ball Position': self.ball_tracker.get_current_position(),
            'Elbow Angle': self.pose_analyzer.get_elbow_angle(),
            'Knee Angle': self.pose_analyzer.get_knee_angle(),
            'Shot Phase': self.pose_analyzer.get_current_phase(),
            'Form Score': self.pose_analyzer.get_form_score()
        }
    
    def is_shot_complete(self) -> bool:
        """Check if a shot has been completed"""
        return self.pose_analyzer.is_shot_complete()
    
    def get_shot_data(self) -> dict:
        """Get data for the completed shot"""
        return {
            'metrics': self.get_current_metrics(),
            'trajectory': self.ball_tracker.get_trajectory().tolist(),
            'predicted_trajectory': self.ball_tracker.predict_trajectory().tolist() if self.ball_tracker.predict_trajectory() is not None else None
        }
    
    def draw_visualizations(self,
                          frame: np.ndarray,
                          ball_center: Optional[np.ndarray],
                          ball_bbox: Optional[np.ndarray],
                          landmarks: Optional[np.ndarray],
                          metrics: dict) -> np.ndarray:
        """Draw analytics visualizations on frame"""
        # Draw ball tracking
        if ball_bbox is not None:
            x1, y1, x2, y2 = map(int, ball_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        if ball_center is not None:
            cv2.circle(frame, tuple(map(int, ball_center)), 5, (0, 255, 255), -1)
        
        # Draw trajectory
        trajectory = self.ball_tracker.get_trajectory()
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                pt1 = tuple(map(int, trajectory[i-1]))
                pt2 = tuple(map(int, trajectory[i]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw predicted trajectory
        predicted_trajectory = self.ball_tracker.predict_trajectory()
        if predicted_trajectory is not None and ball_center is not None:
            for i in range(len(predicted_trajectory)):
                pt1 = tuple(map(int, predicted_trajectory[i]))
                if i > 0:
                    pt2 = tuple(map(int, predicted_trajectory[i-1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2, cv2.LINE_DASHED)
        
        # Draw metrics
        y_offset = 60
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                text = f"{key}: {value:.1f}"
                cv2.putText(frame, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Analysis complete.")

def main():
    parser = argparse.ArgumentParser(description='Basketball Shot Analyzer')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, or video file path)')
    args = parser.parse_args()
    
    # Convert source to int if it's a number (camera index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    try:
        analyzer = BasketballShotAnalyzer(source)
        analyzer.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 