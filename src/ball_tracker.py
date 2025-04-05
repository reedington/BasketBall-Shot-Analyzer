import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from typing import Tuple, Optional, Dict

class BallTracker:
    def __init__(self, model_path: str = 'yolov8n.pt', buffer_size: int = 32):
        """
        Initialize ball tracker
        Args:
            model_path: Path to YOLOv8 model
            buffer_size: Number of frames to keep in trajectory buffer
        """
        print("Initializing YOLOv8 model for ball tracking...")
        self.model = YOLO(model_path)
        self.trajectory_buffer = deque(maxlen=buffer_size)
        self.confidence_threshold = 0.5
        self.current_position = None
        self.current_bbox = None
        self.current_confidence = 0.0
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Detect basketball in frame
        Returns: (center_point, bounding_box, confidence) or (None, None, 0.0)
        """
        results = self.model(frame, classes=[32], conf=self.confidence_threshold)  # 32 is COCO class for sports ball
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get detection with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            if confidences[best_idx] >= self.confidence_threshold:
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                center = np.array([
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ])
                
                self.current_position = center
                self.current_bbox = bbox
                self.current_confidence = confidences[best_idx]
                
                self.trajectory_buffer.append(center)
                return center, bbox, confidences[best_idx]
        
        self.current_position = None
        self.current_bbox = None
        self.current_confidence = 0.0
        return None, None, 0.0
    
    def get_current_position(self) -> Optional[np.ndarray]:
        """Get current ball position"""
        return self.current_position
    
    def get_current_bbox(self) -> Optional[np.ndarray]:
        """Get current ball bounding box"""
        return self.current_bbox
    
    def get_current_confidence(self) -> float:
        """Get current detection confidence"""
        return self.current_confidence
    
    def get_trajectory(self) -> np.ndarray:
        """Get recent ball trajectory points"""
        return np.array(list(self.trajectory_buffer))
    
    def predict_trajectory(self) -> Optional[np.ndarray]:
        """Predict next ball position based on recent trajectory"""
        if len(self.trajectory_buffer) < 3:
            return None
            
        points = np.array(list(self.trajectory_buffer))
        
        # Fit quadratic curve to y-coordinates (accounting for gravity)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Get time steps
        t = np.arange(len(points))
        
        # Fit quadratic for y (gravity) and linear for x
        x_coef = np.polyfit(t, x_coords, 1)
        y_coef = np.polyfit(t, y_coords, 2)
        
        # Predict next 5 positions
        t_pred = np.arange(len(points), len(points) + 5)
        x_pred = np.polyval(x_coef, t_pred)
        y_pred = np.polyval(y_coef, t_pred)
        
        return np.column_stack((x_pred, y_pred)) 