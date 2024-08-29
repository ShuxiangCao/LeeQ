class KalmanFilter1D:
    def __init__(self, initial_position, position_variance):
        # Initial position estimate
        self.x = initial_position

        # Initial variance of the estimate
        self.P = position_variance

    def predict(self, position_variance, movement=0):
        # Update the estimate via movement
        self.x += movement

        # Update the error covariance
        self.P += position_variance

    def update(self, measurement, measurement_variance):
        # Compute the Kalman Gain
        S = self.P + measurement_variance
        K = self.P / S

        # Update the estimate via measurement
        y = measurement - self.x
        self.x += K * y

        # Update the error covariance
        self.P = (1 - K) * self.P

