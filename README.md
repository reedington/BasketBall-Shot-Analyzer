# Basketball Shot Analyzer

A real-time basketball shot analysis tool that uses computer vision to track shooting form and ball trajectory.

## Features

- Real-time pose estimation using MediaPipe
- Basketball tracking using YOLOv8
- Shot form analysis
- Ball trajectory prediction
- Real-time feedback on shooting mechanics

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for better performance)
- Webcam or video file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/basketball-shot-analyzer.git
cd basketball-shot-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using Webcam
```bash
python src/main.py
```

### Using Video File
```bash
python src/main.py --source path/to/video.mp4
```

### Controls
- Press 'q' to quit the application

## Features in Detail

### Pose Analysis
- Tracks body keypoints using MediaPipe
- Analyzes shooting form angles
- Detects shot phases (set point, loading, release, follow-through)

### Ball Tracking
- Real-time basketball detection using YOLOv8
- Ball trajectory tracking
- Trajectory prediction

### Metrics
- Elbow angle
- Knee angle
- Shot phase
- Form score
- FPS counter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 