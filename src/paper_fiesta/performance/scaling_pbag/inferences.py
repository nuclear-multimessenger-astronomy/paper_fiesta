import os
import timeit
import time
import gc

import numpy as np
import jax
import jax.numpy as jnp
from arviz import ess

from fiesta.inference.prior import Uniform, Constraint
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.utils import load_event_data



NUMBER = 3

########
# DATA #
########

data = load_event_data("./injection_pyblastafterglow_gaussian.dat")
trigger_time = 58849
FILTERS = data.keys()

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
epsilon_tot = Constraint(xmin = 0., xmax = 1., naming=["epsilon_tot"])
Gamma0 = Uniform(xmin = 100., xmax=1000., naming=["Gamma0"])

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
              epsilon_tot,
              Gamma0]

prior = ConstrainedPrior(prior_list, conversion_function)


################
# LIKELIHOOD & #
# SAMPLING     #
################

rng_key = jax.random.PRNGKey(3451)

likelihood = EMLikelihood(model,
                          data,
                          trigger_time=58849.,
                          tmin=1e-2, 
                          tmax = 200.0,
                          detection_limit=None,
                          fixed_params={"luminosity_distance": 40.0, "redshift": 0.0}
                          )

outdir = f"./outdir/"

timing = []

for j in [10, 20, 40, 80]:
    rng_key, subkey = jax.random.split(rng_key)

    def setup_fiesta():
        global fiesta
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
                    outdir=outdir,
                    seed=np.random.randint(10_000),
                    systematics_file=f"./systematics_{j}.yaml",
                    precompile=True
    )
    
    def sample():
        fiesta.sample(subkey)
        
    for _ in range(NUMBER):

        compile_time = timeit.timeit('setup_fiesta()', globals=globals(), number=1)
        total_time = timeit.timeit("sample()", globals=globals(), number=1)
        ESS = ess(fiesta.posterior_samples).drop_vars('log_prob').to_array().mean().item()
        with open("./outdir/RTX_6000_timing.txt", "a+") as f:
            f.write(f"{j} {total_time:.6e} {compile_time:.6e} {ESS} \n")
