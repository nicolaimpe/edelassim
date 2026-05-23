import abc
from datetime import datetime

import numpy as np
import xarray as xr
from assimilation import AssimilationProblem, Ensemble, StateVector
from numpy.random import PCG64, Generator

from edelassim.observation_operators import zaitchik
from edelassim.postprocess_surfex.prep import compute_all_members_snow_tickness_and_mass


def point_particle_filter(y: np.ndarray, y_hat: np.ndarray, inv_R_trace: np.ndarray) -> np.ndarray:
    """Vectorized particle filtering separate for each point
    y given in an array of shape (n_particles, n_points)
    R is supposed diagonal so we only use its trace here"""
    innovation = y - y_hat
    likelihood = np.exp(-1 / 2 * (innovation * inv_R_trace * innovation))
    evidence = np.sum(likelihood, axis=0)
    new_weights = likelihood / evidence
    return new_weights


def effective_weigths(weights: xr.DataArray):
    return 1 / (weights**2).sum(dim="member")


def kitagawa_resampling(weights: np.ndarray) -> np.ndarray:
    """Vectorized Kiatagawa resampling algorithm
    weights in the
    """
    random_generator = Generator(PCG64())
    random_draw = random_generator.uniform(size=weights.T.shape)
    bin_low = np.zeros_like(random_draw)
    bin_low[:, 1:] = np.cumsum(weights.T[:, :-1], axis=1)
    bin_low = np.expand_dims(bin_low, axis=1)
    bin_high = np.expand_dims(np.cumsum(weights.T, axis=1), axis=1)
    random_draw_expanded = np.expand_dims(random_draw, axis=2)
    duplication_table = (random_draw_expanded > bin_low) * (random_draw_expanded < bin_high)
    n_duplicated_particles = np.sum(duplication_table, axis=1)
    return n_duplicated_particles.T  # particle to duplicate vector


class Filter:
    """Generic bayesian filter"""

    def __init__(self, assimilation_problem: AssimilationProblem, time_step):
        self.assimilation_problem = assimilation_problem
        self.time_step = time_step

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

    def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime, resampling_algorithm: str):
        super().__init__(assimilation_problem, time_step)
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
    """Gordon 1993 boot, astrap filter with particle duplication"""

    def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime, resampling_algorithm: str):
        super().__init__(assimilation_problem, time_step, resampling_algorithm)

    @property
    def particles(self):
        return self.assimilation_problem.ensemble.as_array()

    def propose_new_particles(self) -> np.ndarray:
        return self.assimilation_problem.model(self.particles)


class PointParticleFilter(SIRParticleFilter):
    def __init__(self, assimilation_problem: AssimilationProblem, time_step: datetime):
        super().__init__(assimilation_problem, time_step, resampling_algorithm="kitagawa")

    def update_weights(self) -> np.ndarray:
        y = self.assimilation_problem.observation
        y_hat = self.assimilation_problem.observation_operator(self.particles)
        inv_R_trace = np.diag(self.assimilation_problem.inverse_observation_error_covariance_matrix)
        new_weights = point_particle_filter(y=y, y_hat=y_hat, inv_R_trace=inv_R_trace)
        self.weights = new_weights


if __name__ == "__main__":
    import logging

    import xarray as xr
    from assimilation import AssimilationProblem, Ensemble, StateVector

    logger = logging.getLogger("logger")
    logging.basicConfig(level=logging.INFO)
    slope_map = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/topography/slope.tif"
    prep_files = []
    folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/grandesrousses250m/assim_viirs_local"
    obs_filename = f"{folder}/OBSERVATIONS_220226H12.nc"
    for i in range(1, 18):
        prep_files.append(f"{folder}/PREP_220226H12_PF_ENS{i}.nc")
    observation = (xr.open_dataset(obs_filename).rename({"xx": "x", "yy": "y"}).stack(n_point=("x", "y"))).data_vars["SCF"]
    obs_time = datetime.strptime(obs_filename.split(".")[0][-9:], "%Y%dH%H")
    # swe_data = compute_all_members_snow_tickness_and_mass(prep_files=prep_files, slope_file=slope_map).data_vars["swe"]
    # swe_data.to_netcdf("swe_data.nc")
    swe_data = xr.open_dataset("swe_data.nc").data_vars["swe"].stack(n_point=("x", "y"))
    swe_ensemble = Ensemble(members=[StateVector(swe_data.sel(member=i).values) for i in range(swe_data.sizes["member"])])
    observation_operator = zaitchik
    dummy_model = lambda x: x
    avg_sigma_obs = 0.15

    inv_R = 1 / avg_sigma_obs**2 * np.eye(swe_data.sizes["n_point"])
    assim_problem = AssimilationProblem(
        state=swe_ensemble,
        observation=observation.values,
        model=dummy_model,
        observation_operator=observation_operator,
        inverse_observation_error_covariance_matrix=inv_R,
    )
    logger.info("Creating particle filter object")
    particle_filter = PointParticleFilter(assimilation_problem=assim_problem, time_step=obs_time)
    logger.info("Run assimilation")
    duplicated_particles_array = particle_filter.run()
    print(duplicated_particles_array)
