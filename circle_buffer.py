import numpy as np


class CircleBuffer:
    """CircleBuffer implements a circular buffer. Also implements buffer access methods.
    """

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0 # tracks the oldest element in the array


    def push(self, e):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = e
        self.position = (self.position + 1) % self.size


    def sample(self, n):
        return np.random.choice(self.buffer, n)