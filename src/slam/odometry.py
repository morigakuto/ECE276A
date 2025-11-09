import numpy as np

class Odometry:
    def __init__(self, initial_pose=(0.0, 0.0, 0.0)):
        self.pose = np.array(initial_pose)

    def update(self, delta_pose):
        self.pose = self._compose_poses(self.pose, np.array(delta_pose))

    def _compose_poses(self, pose1, pose2):
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2

        x_new = x1 + x2 * np.cos(theta1) - y2 * np.sin(theta1)
        y_new = y1 + x2 * np.sin(theta1) + y2 * np.cos(theta1)
        
        theta_new = theta1 + theta2
        
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_new, y_new, theta_new])