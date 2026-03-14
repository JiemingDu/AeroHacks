import time 

class PID: 
    def __init__(self, kp, ki, kd, alpha=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha  