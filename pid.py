class PIDController:
    """
    PID controller with derivative low-pass filter and integral windup clamp.

    Args:
        Kp    : proportional gain
        Ki    : integral gain
        Kd    : derivative gain
        alpha : derivative filter smoothing (0 = heavy, 1 = none). Default 0.2
        max_integral : windup clamp on the integral accumulator. Default 500
    """

    def __init__(self, Kp, Ki, Kd, alpha=0.5, max_integral=500):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.alpha = alpha
        self.max_integral = max_integral

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def compute(self, target, current, dt, measured_rate=None):
        """
        Compute PID output.

        Args:
            target  : desired value (commanded variable)
            current : measured value (controlled variable)
            dt      : time since last call in seconds
            measured_rate : optional d(current)/dt from a physical sensor
                           (e.g. gyroscope). When provided, the derivative
                           term uses this instead of differentiating pixel
                           error, giving cleaner damping.

        Returns:
            float: control output
        """
        if dt <= 0:
            return 0.0

        error = target - current

        # proportional
        p_out = self.Kp * error

        # integral with windup clamp
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_out = self.Ki * self.integral

        # derivative with low-pass filter
        if measured_rate is not None:
            # d(error)/dt = -d(current)/dt when target is ~constant
            raw_derivative = -measured_rate
        else:
            raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (self.alpha * raw_derivative +
                                    (1 - self.alpha) * self.filtered_derivative)
        d_out = self.Kd * self.filtered_derivative

        self.prev_error = error

        return p_out + i_out + d_out

    def reset(self):
        """Reset all internal state. Call on re-arm or mode switch."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0