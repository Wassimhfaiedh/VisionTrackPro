# VisionTrack Pro - Object Detection and Tracking System
# This application processes video streams using YOLOv8 for object detection,
# allows users to define Regions of Interest (ROIs), and provides real-time analytics
# through a Flask-based web interface with SocketIO for streaming.

# -----------------------------------
# Import Required Libraries
# -----------------------------------
import cv2  # OpenCV for video processing and computer vision tasks
import numpy as np  # Numerical operations for image processing
from ultralytics import YOLO  # YOLOv8 model for object detection and tracking
from collections import defaultdict  # Dictionary with default values for counting
import torch  # PyTorch for GPU acceleration
from flask import Flask, render_template, request, jsonify  # Flask for web server
from flask_socketio import SocketIO, emit  # SocketIO for real-time communication
import os  # File system operations
import base64  # Encoding frames for streaming
from io import BytesIO  # Handling in-memory file operations
from werkzeug.utils import secure_filename  # Secure file uploads
from datetime import datetime  # Timestamp handling

# -----------------------------------
# Initialize Flask and SocketIO
# -----------------------------------
app = Flask(__name__)  # Create Flask application instance
socketio = SocketIO(app, async_mode='threading')  # Initialize SocketIO with threading mode

# -----------------------------------
# Setup Upload Directory
# -----------------------------------
# Define directory for uploaded videos and create it if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------
# Configure Device (GPU/CPU)
# -----------------------------------
# Check for GPU availability and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cpu':
    print("Warning: GPU not available, falling back to CPU.")

# -----------------------------------
# Define Detection Classes and Colors
# -----------------------------------
# Map class IDs to names for detected objects
class_names = {0: "Person", 2: "Car", 5: "Bus", 7: "Truck"}
# Specify classes to detect
detect_classes = [0, 2, 5, 7]
# Define BGR colors for drawing ROIs
roi_colors = [
    (0, 255, 0),    # Green
    (0, 255, 255),  # Yellow
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (0, 0, 255),    # Red
    (255, 0, 255),  # Magenta
]

# -----------------------------------
# Video Processor Class
# -----------------------------------
class VideoProcessor:
    """Handles video processing, ROI management, and object detection/tracking."""
    
    def __init__(self, video_path, processor_id):
        """Initialize video processor with video path and processor ID."""
        self.video_path = video_path
        self.processor_id = processor_id
        self.model = YOLO('yolov8s.pt').to(device)  # Load YOLOv8 model
        self.roi_list = []  # List of completed ROIs
        self.current_roi = []  # Points for the current ROI being drawn
        self.roi_completed = False  # Flag for completed ROI
        self.counting = False  # Flag for active object counting
        self.roi_counts = defaultdict(int)  # Object counts per ROI
        self.roi_class_counts = defaultdict(lambda: defaultdict(int))  # Class counts per ROI
        self.track_history = defaultdict(list)  # Tracking history for objects
        self.roi_timestamps = defaultdict(list)  # Timestamps for ROI count updates
        self.roi_selected = False  # Flag for selected ROI
        self.roi_moving = False  # Flag for moving ROI
        self.selected_roi_index = None  # Index of selected ROI
        self.last_mouse_pos = None  # Last mouse position for moving ROI
        self.cap = cv2.VideoCapture(video_path)  # Open video file
        self.running = False  # Flag for processing status
        self._processing_task = None  # Background task for processing

        # Validate video file
        if not self.cap.isOpened():
            print(f"Error: Could not open video {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not open video'})
            return

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.wait_time = int(1000 / self.fps) if self.fps > 0 else 30
        self.ret, self.first_frame = self.cap.read()
        if not self.ret:
            print(f"Error: Could not read first frame from {video_path} for processor {processor_id}")
            self.cap = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not read first frame'})
            return
        
        # Store frame dimensions and emit to client
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        socketio.emit(f'frame_dimensions_{self.processor_id}', {
            'width': self.frame_width,
            'height': self.frame_height
        })

    def add_roi_point(self, x, y):
        """Add a point to the current ROI or select an ROI for moving."""
        x = max(0, min(int(x), self.frame_width - 1))  # Clamp x-coordinate
        y = max(0, min(int(y), self.frame_height - 1))  # Clamp y-coordinate
        print(f"Adding point at ({x}, {y}) for processor {self.processor_id}")
        
        if not self.roi_completed or len(self.roi_list) == 0:
            self.current_roi.append((x, y))  # Add point to current ROI
            if len(self.roi_list) > 0 or len(self.current_roi) >= 1:
                self.counting = True  # Enable counting if ROI exists
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'Point added at ({x}, {y})'})
        else:
            # Check if click is inside an existing ROI for moving
            for i, roi in enumerate(self.roi_list):
                if self.is_inside_roi(x, y, roi):
                    self.roi_selected = True
                    self.roi_moving = True
                    self.selected_roi_index = i
                    self.last_mouse_pos = (x, y)
                    socketio.emit(f'update_status_{self.processor_id}', {'message': f'ROI {i + 1} selected for moving'})
                    return
            self.current_roi.append((x, y))  # Start new ROI
            self.counting = True
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'Point added at ({x}, {y})'})

    def move_roi(self, x, y):
        """Move the selected ROI based on mouse movement."""
        x = max(0, min(int(x), self.frame_width - 1))  # Clamp x-coordinate
        y = max(0, min(int(y), self.frame_height - 1))  # Clamp y-coordinate
        if self.roi_moving and self.last_mouse_pos is not None and self.selected_roi_index is not None:
            dx = x - self.last_mouse_pos[0]  # Calculate movement delta
            dy = y - self.last_mouse_pos[1]
            # Update ROI coordinates
            self.roi_list[self.selected_roi_index] = [(int(px + dx), int(py + dy)) for px, py in self.roi_list[self.selected_roi_index]]
            self.last_mouse_pos = (x, y)
            socketio.emit(f'update_status_{self.processor_id}', {'message': f'ROI {self.selected_roi_index + 1} moving'})

    def stop_moving_roi(self):
        """Stop moving the selected ROI."""
        if self.roi_moving:
            self.roi_moving = False
            self.selected_roi_index = None
            self.last_mouse_pos = None
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'ROI movement stopped'})

    def complete_roi(self):
        """Complete the current ROI if it has at least 3 points."""
        if len(self.current_roi) >= 3:
            self.roi_list.append(self.current_roi.copy())  # Save completed ROI
            self.current_roi = []  # Reset current ROI
            self.roi_completed = True
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'ROI completed'})
            socketio.emit(f'update_counts_{self.processor_id}', {'counts': dict(self.roi_counts)})
        else:
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Need at least 3 points to complete ROI'})

    def remove_last_roi(self):
        """Remove the last completed ROI and its associated data."""
        if self.roi_list:
            self.roi_list.pop()  # Remove last ROI
            self.roi_counts.pop(len(self.roi_list), None)  # Clear counts
            self.roi_timestamps.pop(len(self.roi_list), None)  # Clear timestamps
            self.roi_class_counts.pop(len(self.roi_list), None)  # Clear class counts
            if not self.roi_list:
                self.roi_completed = False  # Reset completion flag
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Last zone removed'})
            socketio.emit(f'update_counts_{self.processor_id}', {'counts': dict(self.roi_counts)})

    def is_inside_roi(self, x, y, roi):
        """Check if a point (x, y) is inside a given ROI."""
        if len(roi) < 3:
            return False
        return cv2.pointPolygonTest(np.array(roi, dtype=np.int32), (int(x), int(y)), False) >= 0

    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw a dashed rectangle around detected objects."""
        x1, y1 = pt1
        x2, y2 = pt2
        dash_length = 10
        gap_length = 5
        # Draw dashed lines for top and bottom
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness, lineType=cv2.LINE_AA)
        # Draw dashed lines for left and right
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness, lineType=cv2.LINE_AA)
        # Draw corner markers
        corner_size = 5
        for pt in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            px, py = pt
            cv2.line(img, (px, py), (px + corner_size * (-1 if px == x2 else 1), py), (0, 0, 0), 1)
            cv2.line(img, (px, py), (px, py + corner_size * (-1 if py == y2 else 1)), (0, 0, 0), 1)

    def process_frame(self, frame):
        """Process a video frame, draw ROIs, and count objects within them."""
        if frame is None:
            return None
        display_frame = frame.copy()

        # Draw current ROI points and connecting lines
        for i, point in enumerate(self.current_roi):
            cv2.circle(display_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  # Draw point
            if i > 0:
                cv2.line(display_frame, (int(self.current_roi[i-1][0]), int(self.current_roi[i-1][1])), 
                         (int(point[0]), int(point[1])), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        if len(self.current_roi) >= 3:
            cv2.line(display_frame, (int(self.current_roi[-1][0]), int(self.current_roi[-1][1])), 
                     (int(self.current_roi[0][0]), int(self.current_roi[0][1])), (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw completed ROIs with labels
        for roi_index, roi in enumerate(self.roi_list):
            color_index = roi_index % len(roi_colors)
            border_color = roi_colors[color_index]
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [np.array(roi, dtype=np.int32)], border_color)  # Fill ROI
            display_frame = cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0)  # Blend overlay
            # Draw dashed ROI borders
            for i in range(len(roi)):
                start = roi[i]
                end = roi[(i + 1) % len(roi)]
                length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                segments = int(length / 10) if length > 0 else 1
                for j in range(segments):
                    if j % 2 == 0:
                        t1 = j / segments
                        t2 = (j + 0.5) / segments
                        pt1 = (int(start[0] + t1 * (end[0] - start[0])), 
                               int(start[1] + t1 * (end[1] - start[1])))
                        pt2 = (int(start[0] + t2 * (end[0] - start[0])), 
                               int(start[1] + t2 * (end[1] - start[1])))
                        cv2.line(display_frame, pt1, pt2, border_color, 2, lineType=cv2.LINE_AA)
            if roi:
                # Add zone label
                roi_array = np.array(roi, dtype=np.int32)
                centroid_x = int(np.mean(roi_array[:, 0]))
                centroid_y = int(np.mean(roi_array[:, 1]))
                label = f"Zone {roi_index + 1}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                rect_width = text_size[0] + 10
                rect_height = text_size[1] + 10
                rect_x = centroid_x - rect_width // 2
                rect_y = centroid_y - rect_height // 2
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), border_color, -1)
                display_frame = cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0)
                cv2.putText(display_frame, label, (rect_x + 5, rect_y + text_size[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Process frame for object detection if counting is enabled
        if self.counting:
            results = self.model.track(frame, persist=True, classes=detect_classes, tracker="bytetrack.yaml", device=device)
            current_ids_per_roi = [set() for _ in self.roi_list]
            current_classes_per_roi = [defaultdict(int) for _ in self.roi_list]  # Track classes per ROI
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    class_name = class_names.get(class_id, "Unknown")
                    for roi_index, roi in enumerate(self.roi_list):
                        if self.is_inside_roi(center_x, center_y, roi):
                            current_ids_per_roi[roi_index].add(track_id)
                            current_classes_per_roi[roi_index][class_name] += 1
                            # Draw bounding box and label
                            self.draw_dashed_rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            object_overlay = display_frame.copy()
                            cv2.rectangle(object_overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                            display_frame = cv2.addWeighted(object_overlay, 0.3, display_frame, 0.7, 0)
                            label = class_name
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            rect_x, rect_y = x1, y1 - 30
                            rect_width = text_size[0] + 10
                            rect_height = text_size[1] + 10
                            cv2.rectangle(display_frame, (rect_x, rect_y), 
                                         (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), -1)
                            cv2.putText(display_frame, label, 
                                       (rect_x + 5, rect_y + text_size[1] + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            break
                # Update counts and timestamps
                for roi_index in range(len(self.roi_list)):
                    count = len(current_ids_per_roi[roi_index])
                    self.roi_counts[roi_index] = count
                    self.roi_class_counts[roi_index] = dict(current_classes_per_roi[roi_index])
                    self.roi_timestamps[roi_index].append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'count': count
                    })
                    if len(self.roi_timestamps[roi_index]) > 5:
                        self.roi_timestamps[roi_index] = self.roi_timestamps[roi_index][-5:]
                socketio.emit(f'update_counts_{self.processor_id}', {'counts': dict(self.roi_counts)})
        return display_frame

    def encode_frame(self, frame):
        """Encode a frame as JPEG for streaming to the client."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def export_zone_data(self):
        """Export ROI data including counts, timestamps, and class counts."""
        zone_data = []
        for roi_index in range(len(self.roi_list)):
            latest_timestamp = self.roi_timestamps[roi_index][-1] if self.roi_timestamps[roi_index] else {'time': 'N/A', 'count': 0}
            zone_data.append({
                'zone': roi_index + 1,
                'latest_time': latest_timestamp['time'],
                'latest_count': latest_timestamp['count'],
                'history': self.roi_timestamps[roi_index].copy(),
                'classes': dict(self.roi_class_counts[roi_index])
            })
        return {
            'processor_id': self.processor_id,
            'zones': zone_data
        }

    def process_video(self):
        """Process and stream video frames, looping when the video ends."""
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            if self.counting:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video by resetting to the first frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print(f"Error: Could not reset video {self.video_path} for processor {self.processor_id}")
                        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not reset video'})
                        break
            else:
                frame = self.first_frame.copy() if self.first_frame is not None else None
            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                frame_data = self.encode_frame(processed_frame)
                socketio.emit(f'video_frame_{self.processor_id}', {'frame': frame_data})
            socketio.sleep(self.wait_time / 1000.0)  # Control frame rate
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._processing_task = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Video processing stopped'})

    def start(self):
        """Start video processing from the beginning."""
        if self._processing_task is not None:
            print(f"Processor {self.processor_id} is already running")
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not reopen video {self.video_path} for processor {self.processor_id}")
            socketio.emit(f'update_status_{self.processor_id}', {'message': 'Error: Could not reopen video'})
            return
        self.counting = True
        self.running = True
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Counting started'})
        self._processing_task = socketio.start_background_task(self.process_video)

    def stop(self):
        """Stop video processing and clear data."""
        self.running = False
        self.counting = False
        self.roi_counts.clear()
        self.roi_class_counts.clear()
        self.roi_timestamps.clear()
        if self.cap:
            self.cap.release()
            self.cap = None
        socketio.emit(f'update_status_{self.processor_id}', {'message': 'Video processing stopped'})
        socketio.emit(f'update_counts_{self.processor_id}', {'counts': dict(self.roi_counts)})

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

# -----------------------------------
# Global Processor Dictionary
# -----------------------------------
processors = {}  # Store active video processors by ID

# -----------------------------------
# Flask Routes
# -----------------------------------
@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/upload/<processor_id>', methods=['POST'])
def upload_video(processor_id):
    """Handle video file uploads for a specific processor."""
    print(f"Received upload request for processor {processor_id}")
    if processor_id not in ['1', '2']:
        print(f"Invalid processor ID: {processor_id}")
        return {'error': 'Invalid processor ID'}, 400
    if f'video{processor_id}' not in request.files:
        print(f"No video file provided for processor {processor_id}")
        return {'error': 'No video file provided'}, 400
    video = request.files[f'video{processor_id}']
    if video.filename == '':
        print(f"No video selected for processor {processor_id}")
        return {'error': 'No video selected'}, 400
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Saving video to {video_path}")
    try:
        video.save(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Invalid video file: {video_path}")
            os.remove(video_path)
            return {'error': 'Invalid video file'}, 400
        cap.release()
        processors[processor_id] = VideoProcessor(video_path, processor_id)
        if processors[processor_id].cap is None:
            print(f"Failed to process video: {video_path}")
            os.remove(video_path)
            return {'error': 'Failed to process video'}, 500
        socketio.start_background_task(processors[processor_id].process_video)
        print(f"Video uploaded successfully for processor {processor_id}")
        return {'message': 'Video uploaded successfully'}
    except Exception as e:
        print(f"Upload error for processor {processor_id}: {str(e)}")
        return {'error': f'Upload failed: {str(e)}'}, 500

@app.route('/export_zone_data/<processor_id>', methods=['GET'])
def export_zone_data(processor_id):
    """Export zone data as JSON for a specific processor."""
    if processor_id not in processors:
        return jsonify({'error': 'Processor not found'}), 404
    zone_data = processors[processor_id].export_zone_data()
    return jsonify(zone_data)

# -----------------------------------
# SocketIO Event Handlers
# -----------------------------------
@socketio.on('add_roi_point')
def handle_add_roi_point(data):
    """Handle adding a point to an ROI."""
    processor_id = data['processor_id']
    x, y = data['x'], data['y']
    if processor_id in processors:
        processors[processor_id].add_roi_point(x, y)

@socketio.on('move_roi')
def handle_move_roi(data):
    """Handle moving an ROI."""
    processor_id = data['processor_id']
    x, y = data['x'], data['y']
    if processor_id in processors:
        processors[processor_id].move_roi(x, y)

@socketio.on('stop_moving_roi')
def handle_stop_moving_roi(data):
    """Handle stopping ROI movement."""
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].stop_moving_roi()

@socketio.on('complete_roi')
def handle_complete_roi(data):
    """Handle completing an ROI."""
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].complete_roi()

@socketio.on('start')
def handle_start(data):
    """Handle starting video processing."""
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].start()

@socketio.on('stop')
def handle_stop(data):
    """Handle stopping video processing."""
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].stop()

@socketio.on('remove_last_roi')
def handle_remove_last_roi(data):
    """Handle removing the last ROI."""
    processor_id = data['processor_id']
    if processor_id in processors:
        processors[processor_id].remove_last_roi()

@socketio.on('export_zone_data')
def handle_export_zone_data(data):
    """Handle exporting zone data via SocketIO."""
    processor_id = data['processor_id']
    if processor_id in processors:
        zone_data = processors[processor_id].export_zone_data()
        socketio.emit(f'export_zone_data_{processor_id}', zone_data)

# -----------------------------------
# Main Application Entry Point
# -----------------------------------
if __name__ == '__main__':
    """Start the Flask-SocketIO server."""
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)