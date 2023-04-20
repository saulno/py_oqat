from abc import ABC

class Config(ABC):
    pass

class ACOConfig(Config):
    def __init__(self, algorithm: str, cycles: int, ants: int, alpha: float, rho: float, tau_max: float, tau_min: float):
        self.algorithm = algorithm
        self.cycles = cycles
        self.ants = ants
        self.alpha = alpha
        self.rho = rho
        self.tau_max = tau_max
        self.tau_min = tau_min