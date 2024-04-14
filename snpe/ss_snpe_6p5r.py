### IMPORT PACKAGES AND ALSO REWRITE FUNCTIONS ###

import torch
import multiprocessing
import os

def get_gpu_count():
    return torch.cuda.device_count()

def get_cpu_count():
    return multiprocessing.cpu_count()

if __name__ == "__main__":
    num_gpus = get_gpu_count()
    num_cpus = get_cpu_count()
    print("Number of GPUs (torch.cuda.device_count()):", num_gpus)
    print("Number of CPUs (multiprocessing.cpu_count()):", num_cpus)

print("Number of CPUs (os.cpu_count()):", os.cpu_count())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np

import joblib
import math
from math import ceil
import random
import contextlib
import torch
import sys
from numpy import fft, ndarray
from scipy.stats import norm
import os
import seaborn as sns
from pyro.infer.mcmc.api import MCMC
from warnings import warn
from torch import Tensor, split, randint, cat
from typing import Any, Callable, Optional, Tuple, Union, Dict
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.auto import tqdm, trange
from pyro.infer.mcmc import HMC, NUTS
from sbi.inference import prepare_for_sbi, SNLE
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.types import Shape, TorchTransform
from sbi.utils.get_nn_models import (likelihood_nn,)
from sbi.samplers.mcmc import SliceSamplerVectorized
from sbi.samplers.mcmc.slice_numpy import MCMCSampler
from sbi.utils import tensor2numpy

def seed_all_backends(seed: Optional[Union[int, Tensor]] = None) -> None:
    if seed is None:
        seed = int(torch.randint(10_000_000, size=(1,)))
    else:
        # Cast Tensor to int (required by math.random since Python 3.11)
        seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)
    
    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

class SliceSampler(MCMCSampler):
    def __init__(self, x, lp_f, max_width=float("inf"), init_width: Union[float, np.ndarray] = 0.05, thin=None, tuning: int = 50, verbose: bool = False,):
        MCMCSampler.__init__(self, x, lp_f, thin, verbose=verbose)
        self.max_width = max_width
        self.init_width = init_width
        self.width = None
        self.tuning = tuning
        
    def _tune_bracket_width(self, rng):
        order = list(range(self.n_dims))
        x = self.x.copy()

        self.width = np.full(self.n_dims, self.init_width)

        tbar = trange(self.tuning, miniters=2, disable=not self.verbose)
        tbar.set_description("Tuning bracket width...")
        for n in tbar:
            # for n in range(int(self.tuning)):
            rng.shuffle(order)
            for i in range(self.n_dims):
                x[i], wi = self._sample_from_conditional(i, x[i], rng)
                self.width[i] += (wi - self.width[i]) / (n + 1)

    def _sample_from_conditional(self, i: int, cxi, rng):
        assert self.width is not None, "Chain not initialized."

        # conditional log prob
        Li = lambda t: self.lp_f(np.concatenate([self.x[:i], [t], self.x[i + 1 :]]))
        wi = self.width[i]

        # sample a slice uniformly
        logu = Li(cxi) + np.log(1.0 - rng.rand())

        # position the bracket randomly around the current sample
        lx = cxi - wi * rng.rand()
        ux = lx + wi
        
        # find lower bracket end
        while Li(lx) >= logu and cxi - lx < self.max_width:
            lx -= wi

        # find upper bracket end
        while Li(ux) >= logu and ux - cxi < self.max_width:
            ux += wi

        # sample uniformly from bracket
        xi = (ux - lx) * rng.rand() + lx

        # if outside slice, reject sample and shrink bracket
        while Li(xi) < logu:
            if xi < cxi:
                lx = xi
            else:
                ux = xi
            xi = (ux - lx) * rng.rand() + lx
       
        return xi, ux - lx
      
def run_fun(SliceSamplerSerial, num_samples, inits, seed, log_prob_fn: Callable, thin: Optional[int] = None, tuning: int = 50, verbose: bool = True, init_width: Union[float, np.ndarray] = 0.01,
            max_width: float = float("inf"), num_workers: int = 1, rng=np.random, show_info: bool = False, logger=sys.stdout) -> np.ndarray:
    np.random.seed(seed)
    posterior_sampler = SliceSampler(inits, lp_f=log_prob_fn, max_width=max_width, init_width=init_width, thin=thin, tuning=tuning, verbose=num_workers == 1 and verbose,)
    
    assert num_samples >= 0, "number of samples can't be negative"

    order = list(range(posterior_sampler.n_dims))
    L_trace = []
    samples = np.empty([int(num_samples), int(posterior_sampler.n_dims)])
    logger = open(os.devnull, "w") if logger is None else logger

    if posterior_sampler.width is None:
        # logger.write('tuning bracket width...\n')
        posterior_sampler._tune_bracket_width(rng)

    tbar = trange(int(num_samples), miniters=10, disable=not posterior_sampler.verbose)
    tbar.set_description("Generating samples")
    for n in tbar:
        # for n in range(int(n_samples)):
        for _ in range(posterior_sampler.thin):
            rng.shuffle(order)

            for i in order:
                posterior_sampler.x[i], _ = posterior_sampler._sample_from_conditional(i, posterior_sampler.x[i], rng)

        samples[n] = posterior_sampler.x.copy()

        posterior_sampler.L = posterior_sampler.lp_f(posterior_sampler.x)
        # logger.write('sample = {0}, log prob = {1:.2}\n'.format(n+1, self.L))

        if show_info:
            L_trace.append(posterior_sampler.L)

    # show trace plot
    if show_info:
        fig, ax = plt.subplots(1, 1)
        ax.plot(L_trace)
        ax.set_ylabel("log probability")
        ax.set_xlabel("samples")
        plt.show(block=False)

    return samples

def run(SliceSamplerSerial, log_prob_fn: Callable, num_samples: int, init_params: np.ndarray, num_chains: int = 1, thin: Optional[int] = None, verbose: bool = True, num_workers: int = 1,) -> np.ndarray:
    num_chains , dim_samples = init_params.shape
    # Generate seeds for workers from current random state.
    seeds = torch.randint(high=1_000_000, size=(num_chains,))
    for seed in seeds:
        seed_all_backends(seed)
    with tqdm_joblib(tqdm(range(num_chains), disable=not verbose, desc=f"""Running {num_chains} MCMC chains with {num_workers} worker{"s" if num_workers>1 else ""}.""", total=num_chains,)):
        all_samples = Parallel(n_jobs=num_workers)(delayed(run_fun)(SliceSamplerSerial, num_samples, initial_params_batch, seed, log_prob_fn)for initial_params_batch, seed in zip(init_params, seeds))
    samples = np.stack(all_samples).astype(np.float32)
    samples = samples.reshape(num_chains, -1, dim_samples)  # chains, samples, dim
    samples = samples[:, :: thin, :]  # thin chains

    # save samples
    return samples

class SliceSamplerSerial:
    def __init__(self, log_prob_fn: Callable, init_params: np.ndarray, num_chains: int = 1, thin: Optional[int] = None, tuning: int = 50, verbose: bool = True, init_width: Union[float, np.ndarray] = 0.01, max_width: float = float("inf"), num_workers: int = 1,):
        self._log_prob_fn = log_prob_fn
        self.x = init_params
        self.num_chains = num_chains
        self.thin = thin
        self.tuning = tuning
        self.verbose = verbose
        self.init_width = init_width
        self.max_width = max_width
        self.n_dims = self.x.size
        self.num_workers = num_workers
        self._samples = None

    def get_samples(self, num_samples: Optional[int] = None, group_by_chain: bool = True) -> np.ndarray:
        if self._samples is None:
            raise ValueError("No samples found from MCMC run.")
        # if not grouped by chain, flatten samples into (all_samples, dim_params)
        if not group_by_chain:
            samples = self._samples.reshape(-1, self._samples.shape[2])
        else:
            samples = self._samples

        # if not specified return all samples
        if num_samples is None:
            return samples
        # otherwise return last num_samples (for each chain when grouped).
        elif group_by_chain:
            return samples[:, -num_samples:, :]
        else:
            return samples[-num_samples:, :]

##############################################################################################################################
        
def _maybe_use_dict_entry(default: Any, key: str, dict_to_check: Dict) -> Any:
    attribute = default if key not in dict_to_check.keys() else dict_to_check[key]
    return attribute

def _get_initial_params(proposal, init_strategy: str, num_chains: int, num_workers: int, show_progress_bars: bool, **kwargs,) -> Tensor: 
    # Build init function
    init_fn = proposal._build_mcmc_init_fn(proposal.proposal, proposal.potential_fn, transform=proposal.theta_transform, init_strategy=init_strategy, **kwargs,)

    # Parallelize inits for resampling only.
    if num_workers > 1 and (init_strategy == "resample" or init_strategy == "sir"):
        def seeded_init_fn(seed):
            torch.manual_seed(seed)
            return init_fn()

        seeds = torch.randint(high=10_000_000, size=(num_chains,))

        # Generate initial params parallelized over num_workers.
        with tqdm_joblib(tqdm(range(num_chains), disable=not show_progress_bars, desc=f"""Generating {num_chains} MCMC inits with {num_workers} workers.""", total=num_chains,)):
            initial_params = torch.cat(Parallel(n_jobs=num_workers)(delayed(seeded_init_fn)(seed) for seed in seeds))
    else:
        initial_params = torch.cat([init_fn() for _ in range(num_chains)])
    return initial_params
    
def _slice_np_mcmc(proposal, num_samples: int, potential_function: Callable, initial_params: Tensor, thin: int, warmup_steps: int, vectorized: bool = False, num_workers: int = 1, init_width: Union[float, ndarray] = 0.01, show_progress_bars: bool = True,) -> Tensor:
    num_chains, dim_samples = initial_params.shape
        
    if not vectorized:
        SliceSamplerMultiChain = SliceSamplerSerial
    else:
        SliceSamplerMultiChain = SliceSamplerVectorized

    posterior_sampler = SliceSamplerMultiChain(init_params=tensor2numpy(initial_params), log_prob_fn=potential_function, num_chains=num_chains, thin=thin, verbose=show_progress_bars, num_workers=num_workers, init_width=init_width,)
    warmup_ = warmup_steps * thin
    num_samples_ = ceil((num_samples * thin) / num_chains)
    # Run mcmc including warmup
    samples = run(posterior_sampler, log_prob_fn=potential_function, num_samples = (warmup_ + num_samples_), init_params = tensor2numpy(initial_params))
    samples = samples[:, warmup_steps:, :]  # discard warmup steps
    samples = torch.from_numpy(samples)  # chains x samples x dim

    # Save posterior sampler.
    proposal._posterior_sampler = posterior_sampler

    # Save sample as potential next init (if init_strategy == 'latest_sample').
    proposal._mcmc_init_params = samples[:, -1, :].reshape(num_chains, dim_samples)

    # Collect samples from all chains.
    samples = samples.reshape(-1, dim_samples)[:num_samples, :]
    assert samples.shape[0] == num_samples
    return samples.type(torch.float32).to(proposal._device)

def sample_my_fun(proposal, sample_shape: Shape = torch.Size(), x: Optional[Tensor] = None, method: Optional[str] = None, thin: Optional[int] = None, warmup_steps: Optional[int] = None, num_chains: Optional[int] = None, init_strategy: Optional[str] = None, init_strategy_parameters: Optional[Dict[str, Any]] = None,
                   init_strategy_num_candidates: Optional[int] = None, mcmc_parameters: Dict = {}, mcmc_method: Optional[str] = None, sample_with: Optional[str] = None, num_workers: Optional[int] = None, show_progress_bars: bool = True,) -> Tensor:
    
    proposal.potential_fn.set_x(proposal._x_else_default_x(x))

    # Replace arguments that were not passed with their default.
    method = proposal.method if method is None else method
    thin = proposal.thin if thin is None else thin
    warmup_steps = proposal.warmup_steps if warmup_steps is None else warmup_steps
    num_chains = proposal.num_chains if num_chains is None else num_chains
    init_strategy = proposal.init_strategy if init_strategy is None else init_strategy
    num_workers = proposal.num_workers if num_workers is None else num_workers
    init_strategy_parameters = (proposal.init_strategy_parameters if init_strategy_parameters is None else init_strategy_parameters)

    if init_strategy_num_candidates is not None:
        warn("""Passing `init_strategy_num_candidates` is deprecated as of sbi v0.19.0. Instead, use e.g.,`init_strategy_parameters={"num_candidate_samples": 1000}`""")
        proposal.init_strategy_parameters["num_candidate_samples"] = (init_strategy_num_candidates)
    if sample_with is not None:
        raise ValueError(f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting `sample_with` is no longer supported. You have to rerun `.build_posterior(sample_with={sample_with}).`")
    if mcmc_method is not None:
        warn("You passed `mcmc_method` to `.sample()`. As of sbi v0.18.0, this is deprecated and will be removed in a future release. Use `method` instead of `mcmc_method`.")
        method = mcmc_method
    if mcmc_parameters:
        warn("You passed `mcmc_parameters` to `.sample()`. As of sbi v0.18.0, this is deprecated and will be removed in a future release. Instead, pass the variable to `.sample()` directly, e.g. `posterior.sample((1,), num_chains=5)`.")
    # The following lines are only for backwards compatibility with sbi v0.17.2 or older.
    m_p = mcmc_parameters  # define to shorten the variable name
    method = _maybe_use_dict_entry(method, "mcmc_method", m_p)
    thin = _maybe_use_dict_entry(thin, "thin", m_p)
    warmup_steps = _maybe_use_dict_entry(warmup_steps, "warmup_steps", m_p)
    num_chains = _maybe_use_dict_entry(num_chains, "num_chains", m_p)
    init_strategy = _maybe_use_dict_entry(init_strategy, "init_strategy", m_p)
    proposal.potential_ = proposal._prepare_potential(method)  # type: ignore

    initial_params = _get_initial_params(proposal, init_strategy, num_chains, num_workers, show_progress_bars, **init_strategy_parameters,)
    num_samples = torch.Size(sample_shape).numel()

    track_gradients = method in ("hmc", "nuts")
    with torch.set_grad_enabled(track_gradients):
        if method in ("slice_np", "slice_np_vectorized"):
            transformed_samples = _slice_np_mcmc(proposal, num_samples=num_samples, potential_function=proposal.potential_, initial_params=initial_params, thin=thin, warmup_steps=warmup_steps, vectorized=(method == "slice_np_vectorized"), num_workers=num_workers, show_progress_bars=show_progress_bars,)
        elif method in ("hmc", "nuts", "slice"):
            transformed_samples = _pyro_mcmc(proposal, num_samples=num_samples, potential_function=proposal.potential_, initial_params=initial_params, mcmc_method=method, thin=thin, warmup_steps=warmup_steps, num_chains=num_chains, show_progress_bars=show_progress_bars,)
        else:
            raise NameError

    samples = proposal.theta_transform.inv(transformed_samples)

    return samples.reshape((*sample_shape, -1))  # type: ignore

#######################################################################################################################################

def simulator_seeded(simulator: Callable, theta: Tensor, seed: int) -> Tensor:
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return simulator(theta)

def simulate_in_batches(simulator: Callable, theta: Tensor, sim_batch_size: int = 1, num_workers: int = 1 , seed: Optional[int] = None, show_progress_bars: bool = True, ) -> Tensor:
    num_sims, *_ = theta.shape
    seed_all_backends(seed)
    if num_sims == 0:
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        batches = split(theta, sim_batch_size, dim=0)
        
        if num_workers != 1:
            batch_seeds = randint(high=1_000_000, size=(len(batches),))
            with tqdm_joblib(tqdm(batches, disable=not show_progress_bars, total = len(batches), desc=f"Running {num_sims} simulations in {len(batches)} batches ({num_workers} cores)",)) as _:
                simulation_outputs = Parallel(n_jobs=num_workers)(delayed(simulator_seeded)(simulator, batch, batch_seed) for batch, batch_seed in zip(batches, batch_seeds))
        else:
            pbar = tqdm(total=num_sims, disable=not show_progress_bars, desc=f"Running {num_sims} simulations.", )
            with pbar:
                simulation_outputs = []
                for batch in batches:
                    simulation_outputs.append(simulator_seeded(simulator, batch, seed))
                    pbar.update(sim_batch_size)
        x = cat(simulation_outputs, dim=0)
    else:
        x = simulator(theta)
    return x

def simulate_for_sbi(round_idx: int, simulator: Callable, proposal: Any, num_simulations: int, num_workers: int = 1, simulation_batch_size: int = 1, seed: Optional[int] = None, show_progress_bar: bool = True)-> Tuple[Tensor, Tensor]:
    if round_idx == 0:
        theta = proposal.sample((num_simulations,))
    else:
        theta = sample_my_fun(proposal, (num_simulations,), num_workers = num_workers, num_chains = 4) # because only in first round proposal is boxuniform, then it is mcmcposterior object
    
    x = simulate_in_batches(simulator=simulator, theta=theta, sim_batch_size=simulation_batch_size, num_workers=num_workers, seed=seed, show_progress_bars=show_progress_bar)
    
    return theta, x

### Parallelisation ###
##### paralellise #####

#Parallelise it
# PART 1: Edited code from original sbi repos. Functions edited: simulate_for_sbi, and all the functions it contains.
#Main changes: defined simulator_seeded globally rather than within simulate_in_batches function, and re-imported torch within simulator_seeded 
#NO changes need to be made in this file

# PART 2: Essentially the unparallelised code with 2 extra arguments in the simulate_for_sbi line

##################################### PART 1 #########################################
import joblib
import contextlib
import random
import torch
import numpy as np

from torch import Tensor, split, randint, cat
from typing import Any, Callable, Optional, Tuple, Union
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.auto import tqdm

import sbi
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

from sbi.inference import SNLE, prepare_for_sbi
# No longer importing simulate_for_sbi from the package, we use the one defined above
from sbi import utils as utils
from sbi import analysis as analysis

def seed_all_backends(seed: Optional[Union[int, Tensor]] = None) -> None:
    if seed is None:
        seed = int(torch.randint(1_000_000, size=(1,)))
    else:
        # Cast Tensor to int (required by math.random since Python 3.11)
        seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore

def simulator_seeded(simulator: Callable, theta: Tensor, seed: int) -> Tensor:
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return simulator(theta)

def simulate_in_batches(simulator: Callable, theta: Tensor, sim_batch_size: int = 1, num_workers: int = 1,
                        seed: Optional[int] = None, show_progress_bars: bool = True, ) -> Tensor:

    num_sims, *_ = theta.shape
    seed_all_backends(seed)

    if num_sims == 0:
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        batches = split(theta, sim_batch_size, dim=0)

        if num_workers != 1:
            batch_seeds = randint(high=1_000_000, size=(len(batches),))

            with tqdm_joblib(
                tqdm(batches, disable=not show_progress_bars,
                     desc=f"Running {num_sims} simulations in {len(batches)} batches.", total=len(batches),)) as _:
                simulation_outputs = Parallel(n_jobs=num_workers)(delayed(simulator_seeded)(simulator, batch, batch_seed)
                    for batch, batch_seed in zip(batches, batch_seeds))
        else:
            pbar = tqdm(total=num_sims, disable=not show_progress_bars, desc=f"Running {num_sims} simulations.", )

            with pbar:
                simulation_outputs = []
                for batch in batches:
                    simulation_outputs.append(simulator_seeded(simulator, batch, seed))
                    pbar.update(sim_batch_size)

        x = cat(simulation_outputs, dim=0)
    else:
        x = simulator(theta)

    return x


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

def simulate_for_sbi(simulator: Callable, proposal: Any, num_simulations: int, num_workers: int = 1,
                     simulation_batch_size: int = 1, seed: Optional[int] = None,
                     show_progress_bar: bool = True, ) -> Tuple[Tensor, Tensor]:

    theta = proposal.sample((num_simulations,))

    x = simulate_in_batches(simulator=simulator, theta=theta, sim_batch_size=simulation_batch_size,
                            num_workers=num_workers, seed=seed, show_progress_bars=show_progress_bar,)

    return theta, x

######################################################################################################

# To parallelise, set number of CPUs to be used. Note: parallelise anything that has num_rounds>2 and/or num_simulations>50
# to see total available CPUs: print(os.cpu_count())

CPUs_to_use = 75

total_CPUs = os.cpu_count()
num_workers = CPUs_to_use - total_CPUs -1
# num_workers = -1 uses all cpus
# num_workers = -2 uses all cpus but one etc https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
if CPUs_to_use > total_CPUs:
    raise ValueError(f"Cannot use more CPUs than are available. Available CPUs: {total_CPUs}")

######### Encode summary statistics for 6 params

###### Simulate
headers = ["k1" , "k2", "k3", "a1", "a2", "a3"]          # parameters to be inferred
num_timesteps = 100

t = np.linspace(0, 100, num_timesteps)

param_dict = {'k1': 246.96291990024542, 'k2': 246.96291990024542, 'k3': 246.96291990024542, 'n1': 5, 'n2': 5, 'n3': 5, 'dm1': 1.143402097500176, 'dm2': 1.143402097500176, 'dm3': 1.143402097500176, 'dp1': 0.7833664565550977, 'dp2': 0.7833664565550977, 'dp3': 0.7833664565550977,
              'a1': 24.78485282457379, 'a2': 24.78485282457379, 'a3': 24.78485282457379, 'g1': 0.024884149937163258, 'g2': 0.024884149937163258, 'g3': 0.024884149937163258, 'b1': 33.82307682700831, 'b2': 33.82307682700831, 'b3': 33.82307682700831}

all_params = 'a1', 'a2', 'a3', 'g1', 'g2', 'g3', 'dm1', 'dm2', 'dm3', 'dp1', 'dp2', 'dp3', 'b1', 'b2', 'b3', 'n1', 'n2', 'n3', 'k1', 'k2', 'k3'
new_param_dict ={}
for param in all_params:
    if param not in headers:
        new_param_dict[param] = param_dict[param]
    elif param in headers:
        new_param_dict[param] = param

true_params = tuple(param_dict[parameter] for parameter in headers)
num_dim = len(true_params)


def model(variables, t, params):
    m1, p1, m2, p2, m3, p3 = variables
    k1,k2,k3, a1, a2, a3=params
    g1 = g2 = g3 = 0.024884149937163258
    n1 = n2 = n3 = 5.
    b1 = b2 = b3 = 33.82307682700831
    dm1 = dm2 = dm3 =1.143402097500176
    dp1 = dp2 = dp3 = 0.7833664565550977
    dm1dt = -dm1*m1 + (a1 / (1 + ((1/k1) * p2)**n1)) + g1
    dp1dt = (b1*m1) - (dp1*p1)
    dm2dt = -dm2*m2 + (a2 / (1 + ((1/k2) * p3)**n2)) + g2
    dp2dt = (b2*m2) - (dp2*p2)
    dm3dt = -dm3*m3 + (a3 / (1 + ((1/k3) * p1)**n3)) + g3
    dp3dt = (b3*m3)-(dp3*p3)
    return np.array([dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt])

def solve_ode(params):
    initial_conditions = np.array([0, 2, 0, 1, 0, 3])
    solution = odeint(model, initial_conditions, t=np.linspace(0,100,100), args=(params,)) # The initial value point should be the first element of this sequence
    return solution

timepoints = int(100)
third = int(timepoints / 3)

from scipy.signal import find_peaks
initial_conditions = np.array([0, 2, 0, 1, 0, 3], dtype=np.float32)

def summarise(solution):
    """Calculate summary statistics
        requires solution to be the result of calling solve_ode[parameters]
        e.g. true_data
        Returns
        -------
        a 1D array of summary statistics containing the mean of solution, the log variance of solution, the autocorrelation at lag 10, cross correlation of mRNA and protein species separately, number of peaks
        """
    mean=np.mean(solution, axis=0)
    var=np.var(solution, axis=0)
    #The autocorrelation coefficient of each time series at lag 10 and lag 5
    #can be altered
    auto_cor=np.corrcoef(solution[10:,:], solution[:-10,:], rowvar=False)[6:,:]
    auto_cor_coef=np.diag(auto_cor)
    cor=np.corrcoef(solution, solution, rowvar=False)[:6,2:8]
    cor_coef=np.diag(cor)
    #returning cross correlation coefficients of mRNA to mRNA species, protein to protein species
    truncated_trajectory=solution[third:timepoints,:]
    peaks,_=zip(*(find_peaks(truncated_trajectory[:,i], height=0) for i in range(6)))
    peak_count=np.array([len(p) for p in peaks])
    return(np.concatenate([mean, var, auto_cor_coef, cor_coef, peak_count.flatten()]))

def find_mean_std(num_samples):
    """a function that returns the mean and standard deviation of the summary statistics of the simulated data, for the number of samples specified"""
#generate an empty np array to store simulated data
    simulated_arr=np.zeros((num_samples, 30))
    for i in range(num_samples):
        params_val=np.concatenate((np.random.default_rng().uniform(0.01,250,3),np.random.default_rng().uniform(20.,40.,3))).flatten()
        simulated_data= odeint(model, initial_conditions, t, args=(params_val,))
        simulated_arr[i,:]=summarise(simulated_data)
    s_std=np.std(simulated_arr, axis=0)
    s_mean=np.mean(simulated_arr, axis=0)
    return(s_mean, s_std)

m1,s1=find_mean_std(10000)

def normalise(sum_stats, mean, std):
    """takes in sum stats of 1 by 30 array, mean and std are both 1 by 30 arrays, normalise data by the empirically calculated mean and std"""
    normalised=(sum_stats-mean)/std
    return(normalised)

def my_simulator_ss(theta):
    initial_conditions = np.array([0, 2, 0, 1, 0, 3], dtype=np.float32)
    solution = odeint(model, initial_conditions, t, args=(theta,))
    summarised_data = summarise(solution)
    normalised_ss=normalise(summarised_data, m1, s1)
    return torch.tensor(normalised_ss, dtype=torch.float32) 

x_o = my_simulator_ss(true_params) # Generate observations,with data from 100 timepoints for each of 6 variables
prior = utils.BoxUniform(
    low=torch.tensor([10**-2,10**-2,10**-2, 20., 20., 20.]),
    high=torch.tensor([250.,250.,250., 40., 40., 40.])
)
simulator, prior = prepare_for_sbi(my_simulator_ss, prior) 

# The specific observation we want to focus the inference on.
num_rounds = 5

num_simulations = 3000
simulation_batch_size = 40

neural_posterior = utils.posterior_nn(
    model="maf", hidden_features=50, num_transforms=3)

posteriors = []
proposal = prior
inference = SNPE(prior=prior, density_estimator=neural_posterior)
for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, simulation_batch_size=simulation_batch_size, num_workers=num_workers)
    # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`
    density_estimator = inference.append_simulations(
        theta, x, proposal = proposal
    ).train()
    posterior = inference.build_posterior(density_estimator, sample_with="mcmc")
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

#Save posterior samples
posterior_samples5r = posterior.sample((5000,), x=x_o)

### Save plot...
_ = analysis.pairplot(
    posterior_samples5r, limits=[[-100, 300], [-100, 300], [-100, 300], [20, 40], [20, 40], [20, 40]], figsize=(6, 6) #It should be close to 247, but since this is a toy example we are not expecting it to work (just checking it runs).
)

plt.savefig(f'{num_dim}p-{num_rounds}*{num_simulations}-P_customlimits.png')

### Save csv...
column_name = ['k1','k2','k3', "a1", "a2", "a3"]
snpess6p5r = pd.DataFrame(data=posterior_samples5r, columns=column_name)
snpess6p5r.to_csv('snpess6p5r.csv')
