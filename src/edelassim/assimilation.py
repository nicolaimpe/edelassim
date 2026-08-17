from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class AssimilationProblemError(Exception):
    pass


@dataclass
class StateVector:
    values: np.ndarray

    @property
    def n_points(self):
        return len(self.values)


@dataclass
class Ensemble:
    members: list[StateVector]

    @property
    def n_member(self):
        return len(self.members)

    @property
    def n_points(self):
        return self.members[0].n_points

    def as_array(self) -> np.ndarray:
        return np.array([member.values for member in self.members])


@dataclass
class AssimilationProblem:
    state: StateVector | Ensemble
    observation: np.ndarray
    observation_operator: Callable
    model: Callable
    prior_covariance_matrix: np.ndarray | None = None
    observation_error_covariance_matrix: np.ndarray | None = None
    inverse_prior_covariance_matrix: np.ndarray | None = None
    inverse_observation_error_covariance_matrix: np.ndarray | None = None

    @property
    def ensemble(self):
        if isinstance(self.state, StateVector):
            raise AssimilationProblemError("Use ensemble attribute only for ensemble assimilation methods.")
        return self.state
