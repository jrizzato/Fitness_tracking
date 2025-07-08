import math

class PositionEstimator:
    def __init__(self, mp_pose):
        self.reference_shoulder_width = 40  # cm, average shoulder width
        self.camera_height = 100  # cm, estimated camera height
        self.prev_positions = []
        self.max_history = 10
        self.mp_pose = mp_pose
        
    def estimate_distance_from_camera(self, landmarks, frame_width, frame_height):
        """Estimate distance from camera based on shoulder width"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calculate pixel distance between shoulders
        shoulder_pixel_width = abs(left_shoulder.x - right_shoulder.x) * frame_width
        
        # Avoid division by zero
        if shoulder_pixel_width == 0:
            return None
            
        # Focal length estimation (this would need calibration for accuracy)
        focal_length = frame_width * 0.8  # Rough estimation
        
        # Distance calculation using similar triangles
        distance = (self.reference_shoulder_width * focal_length) / shoulder_pixel_width
        return distance
    
    def estimate_3d_position(self, landmarks, frame_width, frame_height):
        """Estimate 3D position relative to camera"""
        distance = self.estimate_distance_from_camera(landmarks, frame_width, frame_height)
        
        if distance is None:
            return None
            
        # Get center of mass (approximated by torso center)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        # Convert to world coordinates
        world_x = (center_x - 0.5) * distance * 0.8  # Horizontal offset
        world_y = (center_y - 0.5) * distance * 0.6  # Vertical offset
        world_z = distance
        
        position = {
            'x': world_x,
            'y': world_y, 
            'z': world_z,
            'distance': distance
        }
        
        # Smooth the position
        self.prev_positions.append(position)
        if len(self.prev_positions) > self.max_history:
            self.prev_positions.pop(0)
            
        return self.smooth_position()
    
    def smooth_position(self):
        """Apply smoothing to reduce noise"""
        if not self.prev_positions:
            return None
            
        avg_x = sum(p['x'] for p in self.prev_positions) / len(self.prev_positions)
        avg_y = sum(p['y'] for p in self.prev_positions) / len(self.prev_positions)
        avg_z = sum(p['z'] for p in self.prev_positions) / len(self.prev_positions)
        avg_distance = sum(p['distance'] for p in self.prev_positions) / len(self.prev_positions)
        
        return {
            'x': avg_x,
            'y': avg_y,
            'z': avg_z,
            'distance': avg_distance
        }
    
    def estimate_body_orientation(self, landmarks):
        """Estimate body orientation (facing direction)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calculate shoulder line angle
        shoulder_angle = math.atan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        )
        
        # Convert to degrees
        shoulder_angle_deg = math.degrees(shoulder_angle)
        
        # Determine facing direction
        if abs(shoulder_angle_deg) < 30:
            facing = "Front"
        elif abs(shoulder_angle_deg) > 150:
            facing = "Back"
        elif shoulder_angle_deg > 0:
            facing = "Right Side"
        else:
            facing = "Left Side"
            
        return facing, shoulder_angle_deg
