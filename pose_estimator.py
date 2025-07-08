import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
from threading import Thread
from PIL import Image, ImageTk
import sys
from modules.position_estimator import PositionEstimator
from modules.kalman_filter import KalmanFilter

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to count reps for dumbbell curls
def count_reps(angle, rep_count, state, threshold_down=160, threshold_up=70, tolerance=5):
    """Count the reps based on the arm angle movement with state tracking."""
    if angle > threshold_down - tolerance:
        if state == "up":
            state = "down"
    elif angle < threshold_up + tolerance:
        if state == "down":
            rep_count += 1
            state = "up"

    return rep_count, state

# Function to count push-ups based on elbow angle for both arms
def count_pushups_based_on_elbow_angles(left_elbow_angle, right_elbow_angle, rep_count, state, threshold_down=100, threshold_up=160):
    """Count push-ups based on the elbow angles reaching the down and up positions."""
    if left_elbow_angle < threshold_down and right_elbow_angle < threshold_down:
        if state == "up":
            state = "down"
    elif left_elbow_angle > threshold_up and right_elbow_angle > threshold_up:
        if state == "down":
            rep_count += 1
            state = "up"

    return rep_count, state

# Function to count squats based on knee angle
def count_squats(knee_angle, rep_count, state, threshold_down=90, threshold_up=160):
    """Count squats based on the knee angle."""
    if knee_angle < threshold_down:
        if state == "up":
            state = "down"
    elif knee_angle > threshold_up:
        if state == "down":
            rep_count += 1
            state = "up"

    return rep_count, state

# Function to calculate the horizontal distance between two points (for chest fly)
def calculate_horizontal_distance(a, b):
    """Calculate horizontal distance between two points."""
    return abs(a[0] - b[0])

# Function to count reps for chest fly with Kalman filtering and hysteresis
def count_chest_fly_reps(distance, rep_count, state, kalman_filter, min_threshold=100, max_threshold=250, tolerance=20, post_close_buffer=10):
    """Count reps with Kalman filtering and hysteresis logic."""
    
    # Apply Kalman filtering to the wrist distance
    kalman_filter.predict()
    kalman_filter.update(distance)
    smoothed_distance = kalman_filter.get_state()

    # Rep counting logic with hysteresis and buffer
    if state == "close":
        if smoothed_distance > max_threshold - tolerance:
            state = "open"
    elif state == "open":
        if smoothed_distance < min_threshold + tolerance:
            # Buffer check to confirm the state change is valid (e.g., we don't register false reps)
            if smoothed_distance < (min_threshold + tolerance) * 0.9:
                # Ensure the "close" state holds for several frames to avoid missing reps
                rep_count += 1
                state = "close"
    
    return rep_count, state

def open_log_window():
    """Open a separate window for logs."""
    global log_window, log_text_widget
    if log_window is None or not tk.Toplevel.winfo_exists(log_window):
        log_window = tk.Toplevel()
        log_window.title("Log Window")
        log_window.geometry("600x400")

        log_text_widget = tk.Text(log_window, wrap="word", font=("Helvetica", 10))
        log_text_widget.pack(expand=True, fill=tk.BOTH)

def log_message(message):
    """Log messages to the log window."""
    if log_text_widget:
        try:
            log_text_widget.after(0, lambda: _update_log_widget(message))
        except:
            pass
        
def _update_log_widget(message):
    """Update the log widget."""
    log_text_widget.insert(tk.END, message + "\n")
    log_text_widget.see(tk.END)

# Initialize position estimator
position_estimator = PositionEstimator(mp_pose)

# Initialize variables for chest fly rep counting
chest_fly_rep_count = 0
chest_fly_state = "close"
kalman_filter = KalmanFilter()

# Initialize variables for rep counting
left_rep_count = 0
right_rep_count = 0
pushup_rep_count = 0
squat_rep_count = 0
left_state = "down"
right_state = "down"
pushup_state = "up"
squat_state = "up"
log_text_widget = None
log_window = None
stop_tracking = False
video_path = None

def start_tracking(canvas, info_label):
    global stop_tracking, video_path, chest_fly_state, chest_fly_rep_count, left_rep_count, right_rep_count, pushup_rep_count, squat_rep_count, left_state, right_state, pushup_state, squat_state

    # Use camera if no video is selected
    if not video_path:
        cap = cv2.VideoCapture(0)  # Use webcam
        log_message("Using webcam for tracking.")
    else:
        cap = cv2.VideoCapture(video_path)
        log_message("Using video file for tracking.")
    
    log_message("Tracking started.")
    print(selected_fitness_option.get())
    log_message(f"Started tracking for: {selected_fitness_option.get()}")

    while not stop_tracking:
        ret, frame = cap.read()
        if not ret:
            if video_path:
                log_message("End of video or error reading the frame.")
                break
            else:
                continue

        display_width, display_height = 800, 450
        frame = cv2.resize(frame, (display_width, display_height))
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            landmarks = results.pose_landmarks.landmark

            # Estimate position from camera
            position = position_estimator.estimate_3d_position(landmarks, w, h)
            facing, shoulder_angle = position_estimator.estimate_body_orientation(landmarks)

            # Left arm key points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

            # Right arm key points
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

            # Leg key points for squats
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

            # Calculate horizontal distance for chest fly
            chest_fly_distance = calculate_horizontal_distance(left_wrist, right_wrist)

            # Count chest fly reps with Kalman filter
            chest_fly_rep_count, chest_fly_state = count_chest_fly_reps(
                chest_fly_distance, chest_fly_rep_count, chest_fly_state, kalman_filter)
            
            # Calculate angles for left and right elbows (for push-ups)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Calculate knee angle for squats
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Count repetitions for each exercise
            left_rep_count, left_state = count_reps(left_elbow_angle, left_rep_count, left_state)
            right_rep_count, right_state = count_reps(right_elbow_angle, right_rep_count, right_state)

            # Count push-ups based on elbow angles
            pushup_rep_count, pushup_state = count_pushups_based_on_elbow_angles(left_elbow_angle, right_elbow_angle, pushup_rep_count, pushup_state)

            # Count squats based on knee angle
            squat_rep_count, squat_state = count_squats(left_knee_angle, squat_rep_count, squat_state)

            # Display position information
            if position:
                cv2.putText(frame, f'Distance: {position["distance"]:.1f}cm', (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Position: X:{position["x"]:.1f} Y:{position["y"]:.1f} Z:{position["z"]:.1f}', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Facing: {facing}', (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            try:
                log_message(f"{selected_fitness_option.get()} : {left_rep_count} |")
                if selected_fitness_option.get()=="Dumbbell Count":
                    cv2.putText(frame, f'Dumbbell Left Reps: {left_rep_count}', (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Dumbbell Right Reps: {right_rep_count}', (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    
                elif selected_fitness_option.get()=="Push-up Rep Count":
                    cv2.putText(frame, f'Push-up Reps: {pushup_rep_count}', (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    
                elif selected_fitness_option.get()=="Squat Rep Count":
                    cv2.putText(frame, f'Squat Reps: {squat_rep_count}', (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    
                elif selected_fitness_option.get()== "Chest Fly Rep Count":
                    cv2.putText(frame, f'Chest Fly Reps: {chest_fly_rep_count}', (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    
                else:
                    log_message(f"Invalid Fitness Option")
            except:
                continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use after() to schedule the image update on the main thread
        try:
            canvas.after(0, update_canvas, canvas, ImageTk.PhotoImage(Image.fromarray(img)))
        except:
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_message("Tracking stopped.")

def use_camera():
    """Switch to camera mode"""
    global video_path
    video_path = None
    messagebox.showinfo("Camera Mode", "Camera mode activated. Click 'Start Tracking' to begin.")
    log_message("Switched to camera mode.")

def load_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    if video_path:
        messagebox.showinfo("Video Selected", f"Selected Video: {video_path}")
        log_message(f"Video selected: {video_path}")

def stop_tracking_process():
    global stop_tracking
    stop_tracking = True
    log_message("Tracking stopped by user.")

def stop_monitoring():
    global stop_tracking
    stop_tracking = True
    root.quit()  # Stops the Tkinter main loop
    sys.exit(1)

def update_canvas(canvas, img):
    """Update the canvas with the image."""
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img

def reset_counters():
    """Reset all rep counters"""
    global left_rep_count, right_rep_count, pushup_rep_count, squat_rep_count, chest_fly_rep_count
    global left_state, right_state, pushup_state, squat_state, chest_fly_state
    
    left_rep_count = 0
    right_rep_count = 0
    pushup_rep_count = 0
    squat_rep_count = 0
    chest_fly_rep_count = 0
    left_state = "down"
    right_state = "down"
    pushup_state = "up"
    squat_state = "up"
    chest_fly_state = "close"
    
    log_message("All counters reset.")
    messagebox.showinfo("Reset", "All counters have been reset.")

# Create the Tkinter window
root = tk.Tk()
root.title("Enhanced Fitness Tracking System with Position Estimation")
root.geometry("1000x750")
root.configure(bg="#40E0D0")  # For Turquoise background

# Define a style for the buttons
style = ttk.Style()
style.configure(
    "TButton",
    font=("Helvetica", 12, "bold"),
    foreground="#000000",
    background="#ff0000",
    padding=5)

style.map(
    "TButton",
    background=[('active', '#cc0000')]
)

selected_fitness_option = tk.StringVar()

# Title
title_label = tk.Label(root, text="Enhanced Fitness Tracking with Position Estimation", 
                      font=("Helvetica", 18, "bold"), bg="#f0f8ff", fg="#333")
title_label.pack(pady=10)

# Video canvas
canvas = tk.Canvas(root, width=800, height=450, bg="#dcdcdc")
canvas.pack(pady=10)

# Info label
info_label = tk.Label(root, text="Position and exercise data will appear here during tracking", 
                     font=("Helvetica", 12), bg="#f0f8ff", justify=tk.LEFT, anchor="w")
info_label.pack(pady=5, fill=tk.BOTH, padx=10)

# Input source selection
source_frame = tk.Frame(root, bg="#f0f8ff")
source_frame.pack(pady=5)

tk.Label(source_frame, text="Input Source:", font=("Helvetica", 12, "bold"), bg="#f0f8ff").pack(side=tk.LEFT, padx=5)

camera_button = ttk.Button(source_frame, text="Use Camera", style="TButton", command=use_camera)
camera_button.pack(side=tk.LEFT, padx=5)

load_video_button = ttk.Button(source_frame, text="Load Video File", style="TButton", command=load_video)
load_video_button.pack(side=tk.LEFT, padx=5)

# Dropdown for fitness options
fitness_frame = tk.Frame(root, bg="#f0f8ff")
fitness_frame.pack(pady=5)

fitness_label = tk.Label(fitness_frame, text="Select Exercise:", font=("Helvetica", 12, "bold"), bg="#f0f8ff")
fitness_label.pack(side=tk.LEFT, padx=5)

fitness_options = ["Dumbbell Count", "Push-up Rep Count", "Squat Rep Count", "Chest Fly Rep Count"]
fitness_dropdown = ttk.Combobox(fitness_frame, textvariable=selected_fitness_option, values=fitness_options, state="readonly")
fitness_dropdown.pack(side=tk.LEFT, padx=5)
fitness_dropdown.current(0)

# Control buttons
button_frame = tk.Frame(root, bg="#f0f8ff")
button_frame.pack(pady=10)

start_button = ttk.Button(button_frame, text="Start Tracking", style="TButton",
                          command=lambda: Thread(target=start_tracking, args=(canvas, info_label)).start())
start_button.grid(row=0, column=0, padx=5)

stop_button = ttk.Button(button_frame, text="Stop Tracking", style="TButton", command=stop_tracking_process)
stop_button.grid(row=0, column=1, padx=5)

reset_button = ttk.Button(button_frame, text="Reset Counters", style="TButton", command=reset_counters)
reset_button.grid(row=0, column=2, padx=5)

log_button = ttk.Button(button_frame, text="View Logs", style="TButton", command=open_log_window)
log_button.grid(row=0, column=3, padx=5)

exit_button = ttk.Button(button_frame, text="Exit", style="TButton", command=stop_monitoring)
exit_button.grid(row=0, column=4, padx=5)

# Position info display
position_frame = tk.LabelFrame(root, text="Position Information", font=("Helvetica", 12, "bold"), 
                              bg="#f0f8ff", fg="#333")
position_frame.pack(pady=10, padx=20, fill=tk.X)

position_info = tk.Label(position_frame, text="Position estimation will appear here during tracking", 
                        font=("Helvetica", 10), bg="#f0f8ff", justify=tk.LEFT)
position_info.pack(pady=5)

# Run the Tkinter loop
root.mainloop()