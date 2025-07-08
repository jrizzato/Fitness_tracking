import numpy as np

# Kalman Filter class for smoothing the distance
class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(2)  # Position and velocity
        self.P = np.eye(2) * 1000  # Covariance matrix
        self.F = np.array([[1, 1], [0, 1]])  # State transition matrix
        self.H = np.array([1, 0]).reshape(1, 2)  # Measurement matrix
        self.R = np.array([[5]])  # Measurement noise covariance
        self.Q = np.array([[1, 0], [0, 1]])  # Process noise covariance
        self.B = np.array([[0], [0]])  # Control matrix (not used)

    def predict(self):
        # Prediction step
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        return self.state[0]  # Return the position (smoothed distance)