import numpy as np

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

# Function to calculate the horizontal distance between two points (for chest fly)
def calculate_horizontal_distance(a, b):
    """Calculate horizontal distance between two points."""
    return abs(a[0] - b[0])


if __name__ == "__main__":
    # Example usage
    a = (1, 2)
    b = (3, 4)
    c = (5, 6)

    angle = calculate_angle(a, b, c)
    print(f"Angle between points {a}, {b}, and {c} is: {angle} degrees")

    a = (1, 2)
    b = (4, 2)
    c = (4, 6)

    angle = calculate_angle(a, b, c)
    print(f"Angle between points {a}, {b}, and {c} is: {angle} degrees")

    distance = calculate_horizontal_distance(a, b)
    print(f"Horizontal distance between points {a} and {b} is: {distance}")