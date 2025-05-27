import os

import numpy as np
import jax
import jax.numpy as jnp

from fiesta.inference.prior import Uniform, Constraint
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import BullaLightcurveModel, AfterglowFlux
from fiesta.utils import load_event_data




########
# DATA #
########

data = load_event_data("./data/injection_Bu2025.dat")
trigger_time = 58849
FILTERS = data.keys()

#########
# MODEL #
#########

model = AfterglowFlux(name="Bu2025_MLP",
                      filters = FILTERS)


#########
# PRIOR #
#########

inclination_EM = Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM'])
log10_mej_dyn = Uniform(xmin=-3.0, xmax=-1.30, naming=["log10_mej_dyn"])
v_ej_dyn = Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"])
Ye_dyn = Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"])
log10_mej_wind = Uniform(xmin=-2.0, xmax=-0.89, naming=["log10_mej_wind"])
v_ej_wind = Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"])
Ye_wind = Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
sys_err = Uniform(xmin=0.5, xmax=2.0, naming=["sys_err"])

prior_list = [inclination_EM, 
              log10_mej_dyn,
              v_ej_dyn,
              Ye_dyn,
              log10_mej_wind,
              v_ej_wind,
              Ye_wind,
              sys_err]

prior = ConstrainedPrior(prior_list)

################
# LIKELIHOOD & #
# SAMPLING     #
################
  
  
detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          FILTERS,
                          tmin=1e-2,
                          tmax = 200.0,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 40.0, "redshift": 0.0},
                          )



mass_matrix = jnp.eye(prior.n_dim)
eps = 5e-3
local_sampler_arg = {"step_size": mass_matrix * eps}

# Save for postprocessing
outdir = f"./outdir_fiesta/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

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
                outdir = outdir)

fiesta.sample(jax.random.PRNGKey(92))
fiesta.print_summary()
fiesta.save_results()
fiesta.plot_lightcurves()
fiesta.plot_corner()