import abc
import logging
from collections.abc import Callable
from datetime import datetime

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from numpy.random import PCG64, Generator

from edelassim.assimilation import AssimilationProblem, Ensemble, StateVector


def point_particle_filter(y: np.ndarray, y_hat: np.ndarray, inv_R_trace: np.ndarray) -> np.ndarray:
    """Vectorized particle filtering separate for each point
    y given in an array of shape (n_particles, n_points)
    R is supposed diagonal so we only use its trace here"""
    innovation = y - y_hat
    likelihood = np.exp(-1 / 2 * (innovation * inv_R_trace * innovation))
    evidence = np.sum(likelihood, axis=0)
    new_weights = likelihood / evidence
    return new_weights


def effective_weights_xarray(weights: xr.DataArray):
    return 1 / (weights**2).sum(dim="member")


def effective_weights(weights: np.ndarray):
    return 1 / np.sum(weights**2, axis=0)


def reorder_1d_array(resampling_duplication_point: np.ndarray) -> np.ndarray:
    unique_values, counts = np.unique(resampling_duplication_point, return_counts=True)
    reordered_array = np.zeros_like(resampling_duplication_point)
    reordered_array[unique_values - 1] = unique_values
    leftovers = np.repeat(unique_values[counts > 1], counts[counts > 1] - 1)
    reordered_array[reordered_array != np.arange(1, len(reordered_array) + 1)] = leftovers
    return reordered_array


def reorder_duplicated_particles(resampling_duplication_table: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(reorder_1d_array, 1, resampling_duplication_table)


def kitagawa_resampling(weights: np.ndarray, reorder: bool = True) -> np.ndarray:
    """Vectorized Kiatagawa resampling algorithm
    weights in the
    """
    random_generator = Generator(PCG64())
    random_draw_init = random_generator.uniform(size=(weights.T.shape[0], 1)) / weights.shape[1]
    random_draw = random_draw_init[None, :] + np.arange(start=0, step=1 / 17, stop=1)
    random_draw = random_draw[0]
    bin_low = np.zeros_like(random_draw)
    bin_low[:, 1:] = np.cumsum(weights.T[:, :-1], axis=1)
    bin_low = np.expand_dims(bin_low, axis=1)
    bin_high = np.expand_dims(np.cumsum(weights.T, axis=1), axis=1)
    random_draw_expanded = np.expand_dims(random_draw, axis=2)
    duplication_table = (random_draw_expanded > bin_low) * (random_draw_expanded < bin_high)
    n_duplicated_particles = np.argmax(duplication_table, axis=2) + 1
    if reorder:
        n_duplicated_particles = reorder_duplicated_particles(n_duplicated_particles)
    return n_duplicated_particles  # particle to duplicate vector


class Filter:
    """Generic bayesian filter"""

    def __init__(self, assimilation_problem: AssimilationProblem):
        self.assimilation_problem = assimilation_problem

    @abc.abstractmethod
    def prediction(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update(self) -> np.ndarray:
        pass

    def run(self) -> np.ndarray:
        self.prediction()
        return self.update()


class ParticleFilter(Filter):
    """Abstract Particle filter"""

    def __init__(self, assimilation_problem: AssimilationProblem, resampling_algorithm: str):
        super().__init__(assimilation_problem)
        self.resampling_algorithm = resampling_algorithm
        self.reset_weights()

    def reset_weights(self):
        self.weights = np.ones((self.assimilation_problem.ensemble.n_member, self.assimilation_problem.ensemble.n_points))

    def prediction(self) -> np.ndarray:
        return self.propose_new_particles()

    @abc.abstractmethod
    def propose_new_particles(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update_weights(self):
        pass

    def update(self):
        self.update_weights()
        return self.resampling()

    def resampling(self) -> np.ndarray:
        algorithm_dict = {"kitagawa": kitagawa_resampling}
        return algorithm_dict[self.resampling_algorithm](self.weights)


class SIRParticleFilter(ParticleFilter):
    """Gordon 1993 bootstrastrap filter with particle duplication"""

    def __init__(self, assimilation_problem: AssimilationProblem, resampling_algorithm: str):
        super().__init__(assimilation_problem, resampling_algorithm)

    @property
    def particles(self):
        return self.assimilation_problem.ensemble.as_array()

    def propose_new_particles(self) -> np.ndarray:
        return self.assimilation_problem.model(self.particles)


class PointParticleFilter(SIRParticleFilter):
    def __init__(self, assimilation_problem: AssimilationProblem, inflation: bool = False, min_n_eff: int = 8, min_alpha=0.1):
        super().__init__(assimilation_problem, resampling_algorithm="kitagawa")
        self.inflation = inflation
        self.min_n_eff = min_n_eff
        self.min_alpha = min_alpha

    def inflate(self, old_weights: np.ndarray, inflation_step: float = 0.01):

        y = self.assimilation_problem.observation
        y_hat = self.assimilation_problem.observation_operator(self.particles)

        inv_R_trace = np.diag(self.assimilation_problem.inverse_observation_error_covariance_matrix)
        alpha = np.ones_like(inv_R_trace)
        old_n_eff = effective_weights(old_weights)
        degenerated_mask = old_n_eff < self.min_n_eff

        iterations = 1
        n_eff = old_n_eff
        weights = old_weights
        alpha[degenerated_mask] -= inflation_step
        logger.info("Enter inflation loop")
        while np.sum(degenerated_mask) > 0 and np.min(alpha) > self.min_alpha:
            logger.info(
                f"Iteration no. {iterations}, degenerated points {np.sum(degenerated_mask)}, alpha={np.min(alpha):.1f}, R multiplication factor= {1 / np.min(alpha):.1f}"
            )

            new_weights = point_particle_filter(
                y=y[degenerated_mask],
                y_hat=y_hat[:, degenerated_mask],
                inv_R_trace=inv_R_trace[degenerated_mask] * alpha[degenerated_mask],
            )
            weights[:, degenerated_mask] = new_weights
            n_eff[degenerated_mask] = effective_weights(weights=new_weights)
            degenerated_mask = n_eff < self.min_n_eff
            iterations += 1
            alpha[degenerated_mask] -= inflation_step
        return weights, alpha

    def update_weights(self) -> np.ndarray:
        y = self.assimilation_problem.observation
        y_hat = self.assimilation_problem.observation_operator(self.particles)
        # print(y_hat)
        # print(y_hat.shape)

        # import matplotlib.pyplot as plt

        # plt.imshow(np.reshape(y_hat[0, :], shape=(143, 101)))
        # plt.show()
        # plt.imshow(np.reshape(y, shape=(143, 101)))
        # plt.show()
        inv_R_trace = np.diag(self.assimilation_problem.inverse_observation_error_covariance_matrix)
        new_weights = point_particle_filter(y=y, y_hat=y_hat, inv_R_trace=inv_R_trace)
        n_eff = effective_weights(weights=new_weights)
        # print(n_eff[:100] < self.min_n_eff)
        # print(np.any(n_eff < self.min_n_eff))
        if self.inflation and np.any(n_eff < self.min_n_eff):
            new_weights, alpha = self.inflate(new_weights)
        df = pd.DataFrame(new_weights).T
        df.to_csv("weights_20220226.csv")
        pd.DataFrame(alpha).to_csv("alpha_20220226.csv")
        self.weights = new_weights


# class LocalParticleFilter(SIRParticleFilter):
#     def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime):
#         super().__init__(assimilation_problem, time_step, resampling_algorithm="kitagawa")

#     @abc.abstractmethod
#     def compute_correlations(self):
#         pass

#     @abc.abstractmethod
#     def distance_function(self):
#         pass

#     @abc.abstractmethod
#     def localize(self, field: np.ndarray):
#         return self.distance_function(self.compute_correlations(field))

#     def update_weights(self) -> np.ndarray:
#         y = self.assimilation_problem.observation
#         y_hat = self.assimilation_problem.observation_operator(self.particles)
#         inv_R_trace = np.diag(self.assimilation_problem.inverse_observation_error_covariance_matrix)
#         # new_weights = point_particle_filter(y=y, y_hat=y_hat, inv_R_trace=inv_R_trace)
#         self.weights = new_weights


if __name__ == "__main__":
    import pandas as pd

    from edelassim.observation_operators import zaitchik
    from edelassim.postprocess_surfex.prep import compute_all_members_snow_tickness_and_mass

    logger = logging.getLogger("logger")
    logging.basicConfig(level=logging.INFO)
    slope_map = (
        "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/250m/SLP_GR_L93_250m.tif"
    )
    prep_files = []
    folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/grandesrousses250m/assim_viirs_local"
    obs_filename = f"{folder}/OBSERVATIONS_220226H12.nc"
    for i in range(1, 18):
        prep_files.append(f"{folder}/PREP_220226H12_PF_ENS{i}.nc")
    observation = (xr.open_dataset(obs_filename).rename({"xx": "x", "yy": "y"}).stack(n_point=("y", "x"))).data_vars["SCF"]
    # obs_time = datetime.strptime(obs_filename.split(".")[0][-9:], "%Y%dH%H")
    # swe_data = compute_all_members_snow_tickness_and_mass(prep_files=prep_files, slope_file=slope_map).data_vars["swe"]
    # swe_data.to_netcdf("swe_data.nc")
    swe_data = xr.open_dataset("swe_data.nc").data_vars["swe"]
    swe_data = swe_data.reindex(y=swe_data.coords["y"].sortby("y"), x=swe_data.coords["x"].sortby("x"))
    swe_data = swe_data.stack(n_point=("y", "x"))
    # print(swe_data.isel(n_point=slice(1, 100)).coords["x"])
    # print(swe_data.isel(n_point=slice(1, 100)).coords["y"])

    swe_ensemble = Ensemble(members=[StateVector(swe_data.sel(member=i).values) for i in range(swe_data.sizes["member"])])
    observation_operator = zaitchik
    avg_sigma_obs = 0.15

    inv_R = 1 / avg_sigma_obs**2 * np.eye(swe_data.sizes["n_point"])
    assim_problem = AssimilationProblem(
        state=swe_ensemble,
        observation=observation.values,
        model=lambda x: x,  # dummy model, only interested in update step now
        observation_operator=observation_operator,
        inverse_observation_error_covariance_matrix=inv_R,
    )
    logger.info("Creating particle filter object")
    particle_filter = PointParticleFilter(assimilation_problem=assim_problem, inflation=True)
    logger.info("Run assimilation")
    duplicated_particles_array = particle_filter.update()
    print(duplicated_particles_array[1:20, :])
    df = pd.DataFrame(duplicated_particles_array)
    df.to_csv("part_20220226.csv")
