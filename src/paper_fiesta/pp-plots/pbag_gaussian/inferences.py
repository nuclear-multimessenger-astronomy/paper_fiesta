import os

import numpy as np
import jax
import jax.numpy as jnp

from fiesta.inference.prior import Uniform, Constraint, LogUniform
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.inference.injection import InjectionPyblastafterglow


#########
# MODEL #
#########

FILTERS = ["radio-6GHz", "besselli", "bessellv", "uvot::b", "X-ray-5keV"]
model = AfterglowFlux(name="pbag_gaussian_CVAE",
                      filters = FILTERS)


#########
# PRIOR #
#########

inclination_EM = Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM'])
log10_E0 = Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0'])
thetaCore = Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore'])
alphaWing = Uniform(xmin = 0.2, xmax = 3.5, naming= ["alphaWing"])
thetaWing = Constraint(xmin = 0, xmax = np.pi/2, naming = ["thetaWing"])
log10_n0 = Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0'])
p = Uniform(xmin=2.01, xmax=3.0, naming=['p'])
log10_epsilon_e = Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e'])
log10_epsilon_B = Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B'])
Gamma0 = Uniform(xmin=100., xmax=1000., naming=["Gamma0"])
epsilon_tot = Constraint(xmin = 0., xmax = 1., naming=["epsilon_tot"])
sys_err = Uniform(xmin=0.3, xmax=1.0, naming=["sys_err"])

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

prior_list = [inclination_EM, 
              log10_E0, 
              thetaCore,
              alphaWing,
              log10_n0, 
              p, 
              log10_epsilon_e, 
              log10_epsilon_B,
              thetaWing,
              Gamma0,
              sys_err,
              epsilon_tot]

prior = ConstrainedPrior(prior_list, conversion_function)


################
# LIKELIHOOD & #
# SAMPLING     #
################

rng_key = jax.random.PRNGKey(3451)


quantile_list = []
param_list = []

for j in range(0, 200):
    
    rng_key, subkey = jax.random.split(rng_key)
    param_dict = prior.sample(subkey , n_samples=1)
    param_dict = {key: p.item() for key, p in param_dict.items()}
    param_dict["luminosity_distance"] = 40.0

    injection = InjectionPyblastafterglow(jet_type="gaussian",
                                     filters=FILTERS, 
                                     N_datapoints=75, 
                                     error_budget=0.2, 
                                     tmin=1e-2,
                                     tmax=200, 
                                     trigger_time=58849.)
    
    injection.create_injection(param_dict, file="/work/koehn1/fiesta/fiesta/surrogates/GRB/training_data/pyblastafterglow_gaussian_raw_data.h5")
  
    likelihood = EMLikelihood(model,
                          injection.data,
                          FILTERS,
                          tmin=1e-2, 
                          tmax = 200.0,
                          trigger_time=58849.,
                          detection_limit=None,
                          fixed_params={"luminosity_distance": 40.0, "redshift": 0.0}
                          )

    mass_matrix = jnp.eye(prior.n_dim)
    eps = 5e-3
    local_sampler_arg = {"step_size": mass_matrix * eps}
    
    # Save for postprocessing
    outdir = f"./outdir/"
    
    fiesta = Fiesta(likelihood,
                    prior,
                    n_chains = 1_000,
                    n_loop_training = 7,
                    n_loop_production = 3,
                    num_layers = 4,
                    hidden_size = [64, 64],
                    n_epochs = 20,
                    n_local_steps = 50,
                    n_global_steps = 200,
                    local_sampler_arg=local_sampler_arg,
                    outdir=outdir)
    
    fiesta.sample(jax.random.PRNGKey(42))

    state = fiesta.Sampler.get_sampler_state(training=False)
    chains = state["chains"]
    n_chains, n_steps, n_dim = jnp.shape(chains)
    samples = jnp.reshape(chains, (n_chains * n_steps, n_dim))
    quantiles = [jnp.sum(samples[:,j]<=param_dict[p])/(n_chains * n_steps)  for j, p in enumerate(model.parameter_names)]

    quantile_list.append(quantiles)
    param_list.append([param_dict[p] for p in model.parameter_names])


np.savetxt("./outdir/quantiles.txt", np.array(quantile_list))
np.savetxt("./outdir/params.txt", np.array(param_list))