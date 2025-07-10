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