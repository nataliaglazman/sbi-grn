""" A simple implementation of Approximate Bayesian Inference based on
Sequential Monte Carlo. """

import time
import os
from functools import partial
from random import choices, seed
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore
from p_tqdm import p_umap  # type: ignore
from scipy.stats import multivariate_normal, norm, uniform  # type: ignore

from scoring_9_pars_new_model import score

# Define list of parameters and their prior limits
parlist: List[Dict[str, Union[str, float]]] = [{
    'name': 'log_k1',
    'lower_limit': 10,
    'upper_limit': 250
}, {
    'name': 'log_k2',
    'lower_limit': 10,
    'upper_limit': 250

}, {'name': 'log_k3',
    'lower_limit': 10,
    'upper_limit': 250

}, {'name': 'log_g1',
    'lower_limit': 0,
    'upper_limit': 5

}, {'name': 'log_g2',
    'lower_limit': 0,
    'upper_limit': 5

}, {'name': 'log_g3',
    'lower_limit': 0,
    'upper_limit': 5

}, {'name': 'log_a1',
    'lower_limit': 20,
    'upper_limit': 40

}, {'name': 'log_a2',
    'lower_limit': 20,
    'upper_limit': 40

}, {'name': 'log_a3',
    'lower_limit': 20,
    'upper_limit': 40
}]

def calculate_distance(pars: List[float]) -> float:
    """ The distance function to be optimised. """
    return score_wrapper(*pars)


def score_wrapper(log_k1: float,
                log_k2: float, log_k3: float, log_g1: float, log_g2: float, log_g3: float, log_a1: float, log_a2: float, log_a3: float) -> float:
    """Wrapper function repressilator model with 4 parameters, to be called by the optimiser."""
    #pylint: disable=too-many-arguments

    # Make a parameter dictionary, converting the log-spaced system params
    par_dict = {
        "k1": log_k1,
        "k2": log_k2,
        "k3": log_k3,
        "g1": log_g1,
        "g2": log_g2,
        "g3": log_g3,
        "a1": log_a1,
        "a2": log_a2,
        "a3": log_a3
    }

    # Call the actual scoring function
    return score(par_dict)


################################################################################


def make_output_folder(name: str = "smc_9_pars_new_model") -> None:
    """Make sure the output folder exists, else make it."""
    if not os.path.isdir(name):
        os.mkdir(name)


def sample_prior() -> List[float]:
    """ Generate one random draw of parameters from the priors. """
    prior = []
    for par_entry in parlist:
        keys = par_entry.keys()
        # If limits are given, we use a uniform distribution
        if "lower_limit" in keys and "upper_limit" in keys:
            lower = float(par_entry["lower_limit"])
            upper = float(par_entry["upper_limit"])
            # Note that scale parameter denotes the width of the distribution!
            # docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
            prior.append(uniform.rvs(loc=lower, scale=upper - lower))

        # If mean and stdev are given, we use a Gaussian
        elif "mean" in keys and "stdev" in keys:
            mean = float(par_entry["mean"])
            stdev = float(par_entry["stdev"])
            prior.append(norm.rvs(loc=mean, scale=stdev))

        else:
            raise KeyError("Prior unclear.")
    return prior


def evaluate_parametrisation(pars: List[float]) -> float:
    """ Returns how probably a given system parametrisation is based on
    the specified priors."""
    probability = 1.0
    for par, par_entry in zip(pars, parlist):
        keys = par_entry.keys()
        if "lower_limit" in keys and "upper_limit" in keys:
            lower = float(par_entry["lower_limit"])
            upper = float(par_entry["upper_limit"])
            probability *= uniform.pdf(par, loc=lower, scale=upper - lower)
        elif "mean" in keys and "stdev" in keys:
            mean = float(par_entry["mean"])
            stdev = float(par_entry["stdev"])
            probability *= norm.pdf(par, loc=mean, scale=stdev)
        else:
            raise KeyError("Prior unclear.")
    return probability


def generate_parametrisation(processcall: Any = 0,
                             prev_parametrisations=None,
                             prev_weights=None,
                             eps_dist=10000.0,
                             kernel=None):
    """ Generate one valid parametrisation given a set of previous
    parametrisations and their corresponding weights. The proposed new
    parametrisation will fall under the threshold eps_dist."""
    # pylint: disable=unused-argument
    # processall is a dummy variable for tracking function performance and
    # for using p_tqdm. """

    # Setting random seeds for every thread/process to avoid having the same
    # random sequence in each thread, based on both thread ID and system time [ns]
    # TODO: This is extremely dirty for now, but hopefully it works...
    np.random.seed(processcall)  # type: ignore
    rndint = np.random.randint(low=0, high=1e7)
    
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    # TODO: Use "kernel.random_state(rndint+timeseed)" here for more elegance :)

    # On our way to the next parametriation, we here store the distances
    # of all drawn parametrisations which were deemed possible
    # under the specified priors
    evaluated_distances = []

    # Initialise distance beyond the threshold
    current_dist = eps_dist + 1

    # Find our parametrisation:
    # If we are in the first SMC step:
    if prev_parametrisations is None:
        # We want to find a parametrisatoin
        # with distance below threshold by doing:
        while current_dist > eps_dist:
            # Sample randomly from the prior
            proposed_pars = sample_prior()

            # Check whether the proposed parametrisation
            # is even possible under the priors
            if evaluate_parametrisation(proposed_pars) > 0:
                # if so, store the distance/"score" of it
                current_dist = calculate_distance(proposed_pars)
                evaluated_distances.append(current_dist)

        # Once we got our parametrisation, set its weight to unity
        current_weight = 1.0

    # If we are not in the first SMC step anymore:
    else:
        # we do until we find a parametrisation below threshold:
        while current_dist > eps_dist:
            # propose a parametrisation from the previous parametrisations
            selected_pars = choices(prev_parametrisations,
                                    weights=prev_weights)[0]

            # perturb it using the constructed kernel
            # we need this re-seeding to avoid duplicates           
            timeseed = time.time_ns() % 2**16
            proposed_pars = selected_pars + kernel.rvs(random_state=rndint+timeseed)

            # Check whether the proposed parametrisation
            # is even possible under the priors
            if evaluate_parametrisation(proposed_pars) > 0:
                # if so, store the distance/"score" of it
                current_dist = calculate_distance(proposed_pars)
                evaluated_distances.append(current_dist)

        # Once we got our parametrisation, the weight is calculated
        # in relation to all previously found parametrisations:
        # The weight is given by the probability of the parametrisation
        # based on the priors in the numerator.
        # This is then divided by a sum term over all previously
        # found parametrisations. The sum consists of products of
        # the weight times the probability of the distance between current
        # and previous parametrisation given the kernel.
        sum_denom = 0
        for parametrisation, weight in zip(prev_parametrisations,
                                           prev_weights):
            sum_denom += weight * kernel.pdf(proposed_pars - parametrisation)
        current_weight = evaluate_parametrisation(proposed_pars) / sum_denom

    # Return the proposed parametrisation, its distance, its weight
    # and all the distances we encountered on our way there.
    return proposed_pars, current_dist, current_weight, evaluated_distances


def generate_parametrisations(prev_parametrisations=None,
                              prev_weights=None,
                              eps_dist: float = 10000,
                              n_pars: int = 2000,
                              kernel_factor: float = 1.0):
    """ Call generate_parametrisation() in parallel until n_pars
    parametrisations have been accepted."""

    # If we sampled some parametrisations before, we construct a
    # multivariate-normal kernel based on their covariance matrix.
    # This is used to judge the distance of a new parametrisation from
    # the previously found parametrisations see generation_parametrisation()
    if prev_parametrisations is not None:
        previous_covar = 0.5 * kernel_factor * np.cov(
            np.array(prev_parametrisations).T)
        kernel = multivariate_normal(cov=previous_covar)
    else:
        kernel = None

    # The actual (parallel) call to generate_parametrisation()
    results = p_umap(
        partial(generate_parametrisation,
                prev_parametrisations=prev_parametrisations,
                prev_weights=prev_weights,
                eps_dist=eps_dist,
                kernel=kernel), range(n_pars))

    # All the new parametrisations which have been found, ...
    new_parametrisations = [result[0] for result in results]
    # ... their corresponding distances, ...
    accepted_distances = [result[1] for result in results]
    # ... and corresponding weights.
    new_weights = [result[2] for result in results]
    # And the flattened list of lists of all evaluated distances on the way
    evaluated_distances = [res for result in results for res in result[3]]

    # Print stats of the current run
    new_weights /= np.sum(new_weights)  #Normalising weights
    acceptance_rate = n_pars / len(evaluated_distances)
    print("Acceptance rate:", acceptance_rate)
    print("Min accepted distance: ", np.min(accepted_distances))
    print("Median accepted distance: ", np.median(accepted_distances))
    print("Median evaluated distance: ", np.median(evaluated_distances))
    print("--------------------\n")

    # Return the generated set of new parametrisations from this run
    return new_parametrisations, new_weights, accepted_distances, acceptance_rate


def sequential_abc(initial_dist: float = 700.0,
                   final_dist: float = 5,
                   n_pars: int = 1000,
                   prior_label: Optional[int] = None):
    """ The main function. The sequence of acceptance thresholds starts
    with initial_dist and keeps on reducing until a final threshold
    final_dist is reached.
    prior_label can be used to restart sampling from a previous prior
    distribution in case further exploration with a lower epsilon is needed."""

    # Initialisiation
    make_output_folder()
    distance = initial_dist
    not_converged = True
    last_round = False
    kernelfactor = 1.0

    if prior_label is None:
        # Start from scratch.
        pars = None
        weights = None
        iteration = 0
    else:
        # A file with the label is used to load the posterior.
        # Always use a numerical label, never 'final'
        pars = np.loadtxt(f'smc_9_pars_new_model/pars_{prior_label}.out')
        weights = np.loadtxt(f'smc_9_pars_new_model/weights_{prior_label}.out')
        accepted_distances = np.loadtxt(f'smc_9_pars_new_model/distances_{prior_label}.out')
        distance = np.min(accepted_distances) + \
         0.95*(np.median(accepted_distances) - np.min(accepted_distances))  # type: ignore
        iteration = prior_label

    while not_converged:
        # Perform one iteration step
        iteration += 1
        print(f"SMC step {iteration} with target distance: {distance}")
        pars, weights, accepted_distances, _ = generate_parametrisations(
            prev_parametrisations=pars,
            prev_weights=weights,
            eps_dist=distance,
            n_pars=n_pars,
            kernel_factor=kernelfactor)

        # Propose a new target distance for the subsequent step
        proposed_dist = np.min(accepted_distances) + 0.95 * (
            np.median(accepted_distances) - np.min(accepted_distances))

        # Check whether this was the final round (= we converged the last time)
        if last_round is True:
            # If so, set label accordingly, ...
            label = 'final'
            # ... break loop once we are back at the top
            not_converged = False
        else:
            # Else, the label is just the number of the iteration step.
            label = str(iteration)

        # Write results of the current step to HDD
        np.savetxt(f'smc_9_pars_new_model/pars_{label}.out', pars)  # type: ignore
        np.savetxt(f'smc_9_pars_new_model/weights_{label}.out', weights)  # type: ignore
        np.savetxt(f'smc_9_pars_new_model/distances_{label}.out', accepted_distances)
        
        # Check for convergence, defined as the proposed distance being
        # smaller than the desired final distance.
        if proposed_dist < final_dist:
            # If so, we want to perform one last iteration step
            # with the final distance
            distance = final_dist
            last_round = True
        else:
            # If not, we just continue with the proposed distance
            distance = proposed_dist  # type: ignore

    print('ABC converged succesfully!\n')


if __name__ == "__main__":
    sequential_abc()
