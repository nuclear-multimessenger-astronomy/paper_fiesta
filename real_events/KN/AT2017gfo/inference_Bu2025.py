import os

import numpy as np
import jax
import jax.numpy as jnp


from fiesta.inference.prior import Uniform, Constraint
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import AfterglowFlux, BullaFlux, CombinedSurrogate
from fiesta.utils import load_event_data




########
# DATA #
########

data = load_event_data("../../data/AT2017gfo.dat")
trigger_time = 57982.52851852



FILTERS = list(data.keys())

for key in data.keys():
    mask = (data[key][:,0]-trigger_time >=0.) & (data[key][:,0]-trigger_time <=15.)
    data[key] = data[key][mask]


#########
# MODEL #
#########

model = BullaFlux(name="Bu2025_MLP",
                              filters = FILTERS)

KN_prior = [
            Uniform(xmin=0., xmax=np.pi/2, naming=["inclination_EM"]),
            Uniform(xmin=-3.0, xmax=-1.3, naming=["log10_mej_dyn"]),
            Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"]),
            Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
            Uniform(xmin=-2., xmax=-0.886, naming=["log10_mej_wind"]),
            Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
            Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
]

prior = ConstrainedPrior(KN_prior)



################
# LIKELIHOOD & #
# SAMPLING     #
################
  
  
detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          FILTERS,
                          tmin=0.3,
                          tmax=28.,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 43.583656, "redshift":0.009727})


# Save for postprocessing
outdir = f"./Bu2025/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

fiesta = Fiesta(likelihood,
                prior,
                #error_budget=1.,
                systematics_file="./systematics_file.yaml",
                n_chains = 1_000,
                n_loop_training = 7,
                n_loop_production = 3,
                num_layers = 4,
                hidden_size = [64, 64],
                n_epochs = 20,
                n_local_steps = 50,
                n_global_steps = 200,
                outdir = outdir)

fiesta.sample(jax.random.PRNGKey(42))
fiesta.print_summary()
fiesta.save_results()
fiesta.plot_lightcurves()
fiesta.plot_corner()