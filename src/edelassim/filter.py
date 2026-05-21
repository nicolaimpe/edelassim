import abc
import datetime

import numpy as np
import xarray as xr

from edelassim.assimilation import AssimilationProblem, Ensemble, StateVector

# def vector_particle_filter(y: np.ndarray, y_hat: np.ndarray, inv_R_trace: np.ndarray):
#     """Vectorized particle filtering
#     y given in an array of shape (n_particles, n_points)
#     R is supposed diagonal so we only use its trace here"""
#     innovation = y - y_hat
#     likelihood = np.exp(-1 / 2 * (innovation * inv_R_trace * innovation))
#     evidence = np.sum(likelihood, ax=0)
#     new_weights = likelihood / evidence
#     return new_weights


def point_particle_filter(y: np.ndarray, y_hat: np.ndarray, inv_R_trace: np.ndarray):
    """Vectorized particle filtering separate for each point
    y given in an array of shape (n_particles, n_points)
    R is supposed diagonal so we only use its trace here"""
    innovation = y - y_hat
    likelihood = np.exp(-1 / 2 * (innovation * inv_R_trace * innovation))
    evidence = np.sum(likelihood, ax=0)
    new_weights = likelihood / evidence
    return new_weights


def effective_weigths(weights: xr.DataArray):
    return 1 / (weights**2).sum(dim="member")

def kitagawa_resampling(weights: np.ndarray)->np.ndarray
    return None # particle to duplicate vector

class Filter(abc.ABCMeta):
    """Generic bayesian filter"""

    def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime):
        self.assimilation_problem = assimilation_problem
        self.time_step = time_step

    @abc.abstractmethod
    def prediction(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _perform_analysis(self) -> np.ndarray:
        pass

    def update(self, observation_time_step: datetime) -> np.ndarray:
        new_state = self._perform_analysis()
        self.time_step = observation_time_step
        return new_state

    def run(self) -> np.ndarray:
        self.prediction()
        return self.update()


class ParticleFilter(Filter):
    """Abstract Particle filter"""

    def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime):
        super().__init__(assimilation_problem, time_step)

    def prediction(self) -> np.ndarray:
        return self.propose_new_particles()

    def _perform_analysis(self) -> np.ndarray:
        self.update_weights()
        return self.resampling()

    @abc.abstractmethod
    def propose_new_particles(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update_weights(self):
        pass

    @abc.abstractmethod
    def resampling(self) -> np.ndarray:
        pass


class SIRParticleFilter(Filter):
    """Gordon 1993 bootstrap filter with particle duplication"""

    def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime):
        super().__init__(assimilation_problem, time_step)
        self.reset_weights()

    @abc.abstractmethod
    def resample_algorithm(self) -> np.ndarray:
        pass

    @property
    def particles(self):
        return self.assimilation_problem.ensemble.as_array()

    def reset_weights(self):
        self.weights = np.ones((self.assimilation_problem.ensemble.n_member, self.assimilation_problem.ensemble.n_points))

    def propose_new_particles(self) -> np.ndarray:
        return self.assimilation_problem.model(self.particles)

    def resampling(self) -> np.ndarray:
        duplicated_particles = self.resample_algorithm()
        self.reset_weights()
        return duplicated_particles


class PointParticleFilter(SIRParticleFilter):
    def __init__(self, assimilation_problem, time_step):
        super().__init__(assimilation_problem, time_step)

    def update_weights(self) -> np.ndarray:
        y = self.assimilation_problem.observation
        y_hat = self.assimilation_problem.observation_operator(self.particles)
        inv_R_trace = np.diag(self.assimilation_problem.inverse_observation_error_covariance_matrix)
        return point_particle_filter(y=y, y_hat=y_hat, inv_R_trace=inv_R_trace)

    def resample_algorithm(self) -> np.ndarray:
        return kitagawa_resampling(self.weights)
