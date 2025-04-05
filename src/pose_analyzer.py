import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort

class PoseAnalyzer:
    def __init__(self):
        """Initialize pose analyzer with MediaPipe and DeepSORT tracker"""
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        
        # Initialize shooting form analysis parameters
        self.shooting_phases = {
            'SET_POINT': 0,
            'LOADING': 1,
            'RELEASE': 2,
            'FOLLOW_THROUGH': 3
        }
        
        self.current_phase = None
        self.phase_data = {}
        
        # Current metrics
        self.current_elbow_angle = 0.0
        self.current_knee_angle = 0.0
        self.current_form_score = 0.0
    
    def detect_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose in frame using MediaPipe
        Returns: (landmarks, frame_with_pose)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            frame_with_pose = frame.copy()
            self.mp_drawing.draw_landmarks(
                frame_with_pose,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                                for lm in results.pose_landmarks.landmark])
            
            return landmarks, frame_with_pose
        
        return None, frame
    
    def analyze_pose(self, 
                    frame: np.ndarray,
                    landmarks: Optional[np.ndarray] = None,
                    ball_pos: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze pose for shooting form
        Args:
            frame: Video frame
            landmarks: Pose landmarks if already detected
            ball_pos: Ball position if available
        Returns:
            Dictionary containing pose metrics
        """
        metrics = {}
        
        # Detect pose if not provided
        if landmarks is None:
            landmarks, _ = self.detect_pose(frame)
        
        if landmarks is None:
            return metrics
        
        # Extract relevant joint angles
        self.current_elbow_angle = self._calculate_elbow_angle(landmarks)
        self.current_knee_angle = self._calculate_knee_angle(landmarks)
        
        # Detect shooting phase
        self.current_phase = self._detect_shot_phase(landmarks, ball_pos)
        
        # Calculate form score
        self.current_form_score = self._calculate_form_score(
            self.current_elbow_angle,
            self.current_knee_angle
        )
        
        # Update metrics
        metrics.update({
            'elbow_angle': self.current_elbow_angle,
            'knee_angle': self.current_knee_angle,
            'shot_phase': self.current_phase,
            'form_score': self.current_form_score
        })
        
        return metrics
    
    def get_elbow_angle(self) -> float:
        """Get current elbow angle"""
        return self.current_elbow_angle
    
    def get_knee_angle(self) -> float:
        """Get current knee angle"""
        return self.current_knee_angle
    
    def get_current_phase(self) -> str:
        """Get current shot phase"""
        return self.current_phase or "UNKNOWN"
    
    def get_form_score(self) -> float:
        """Get current form score"""
        return self.current_form_score
    
    def is_shot_complete(self) -> bool:
        """Check if a shot has been completed"""
        return self.current_phase == "FOLLOW_THROUGH"
    
    def _calculate_elbow_angle(self, landmarks: np.ndarray) -> float:
        """Calculate shooting arm elbow angle"""
        # MediaPipe indices for right arm
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        return self._calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
    
    def _calculate_knee_angle(self, landmarks: np.ndarray) -> float:
        """Calculate knee angle for shot alignment"""
        # MediaPipe indices for right leg
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        return self._calculate_angle(hip[:2], knee[:2], ankle[:2])
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _detect_shot_phase(self, 
                          landmarks: np.ndarray,
                          ball_pos: Optional[np.ndarray]) -> str:
        """Detect current phase of shooting motion"""
        wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value][:2]
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:2]
        
        if ball_pos is None:
            return "UNKNOWN"
        
        # Basic phase detection logic
        if wrist[1] > shoulder[1]:  # Wrist below shoulder
            return "SET_POINT"
        elif wrist[1] < shoulder[1] - 0.1:  # Wrist significantly above shoulder
            return "RELEASE"
        else:
            return "LOADING"
    
    def _calculate_form_score(self, elbow_angle: float, knee_angle: float) -> float:
        """Calculate overall form score"""
        # Ideal angles based on professional shooting form
        IDEAL_ELBOW_ANGLE = 90.0
        IDEAL_KNEE_ANGLE = 120.0
        
        elbow_score = 100 - min(abs(elbow_angle - IDEAL_ELBOW_ANGLE) * 2, 100)
        knee_score = 100 - min(abs(knee_angle - IDEAL_KNEE_ANGLE) * 2, 100)
        
        return (elbow_score * 0.6 + knee_score * 0.4) 