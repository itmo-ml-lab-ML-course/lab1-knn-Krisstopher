from enum import Enum

import numpy as np


class Kernel(Enum):
    UNIFORM = lambda dist: np.where(-1 < dist < 1, 0.5, 0)
    GAUSSIAN = lambda dist: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (dist ** 2))
    TRIANGLE = lambda dist: np.max(1 - np.abs(dist), 0)
    EPANECHIKOV = lambda dist: np.max(0.75 * (1 - dist ** 2), 0)

