import os

import numpy as np
import jax
import jax.numpy as jnp


from fiesta.inference.prior import Uniform, Constraint
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import AfterglowFlux, BullaLightcurveModel, CombinedSurrogate
from fiesta.utils import load_event_data




########
# DATA #
########

data = load_event_data("../../data/GRB211211A.dat")
trigger_time = 59559.54791666667
FILTERS = list(data.keys())

#########
# MODEL #
#########

model1 = AfterglowFlux(name="afgpy_gaussian",
                      directory="../../../surrogates/afgpy_gaussian_CVAE/",
                      filters = FILTERS)

model2 = BullaLightcurveModel(name="Bu2024",
                              directory="../../../surrogates/Bu2024_lc/",
                              filters = FILTERS)

model = CombinedSurrogate(models=[model1, model2],
                          sample_times=jnp.logspace(-4, jnp.log10(150), 120))

theta = {"inclination_EM": 0.1,
         "log10_E0": 51.0,
         "thetaCore": 0.1,
         "alphaWing": 2.,
         "log10_n0": -3.,
         "p": 2.5,
         "log10_epsilon_e": -3,
         "log10_epsilon_B": -5.,
         "log10_mej_dyn": -2.,
         "v_ej_dyn": 0.14,
         "Ye_dyn": 0.23,
         "log10_mej_wind": -1.1,
         "v_ej_wind": 0.09,
         "luminosity_distance": 358.47968, 
         "redshift": 0.0763}



#########
# PRIOR #
#########

GRB_prior = [
            Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM']),
             Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0']), 
             Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore']),
             Uniform(xmin = 0.2, xmax=3.5, naming= ["alphaWing"]),
             Constraint(xmin = 0, xmax=np.pi/2, naming = ["thetaWing"]),
             Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0']),
             Uniform(xmin=2.01, xmax=3.0, naming=['p']),
             Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e']),
             Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B']),
             Constraint(xmin = 0., xmax=1., naming=["epsilon_tot"])
]

KN_prior = [
            Uniform(xmin=-3.0, xmax=-1.7, naming=["log10_mej_dyn"]),
            Uniform(xmin=0.12, xmax=0.25, naming=["v_ej_dyn"]),
            Uniform(xmin=0.15, xmax=0.3, naming=["Ye_dyn"]),
            Uniform(xmin=-2., xmax=-0.886, naming=["log10_mej_wind"]),
            Uniform(xmin=0.03, xmax=0.15, naming=["v_ej_wind"])
]



def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

prior_list = [*GRB_prior,
              *KN_prior]

prior = ConstrainedPrior(prior_list, conversion_function)

################
# LIKELIHOOD & #
# SAMPLING     #
################
  
  
detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          FILTERS,
                          tmin=1e-4,
                          tmax=150.,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 358.47968, "redshift": 0.0763})



mass_matrix = jnp.eye(prior.n_dim)
eps = 5e-3
local_sampler_arg = {"step_size": mass_matrix * eps}

# Save for postprocessing
outdir = f"./afgpy/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

fiesta = Fiesta(likelihood,
                prior,
                systematics_file="./systematics_file.yaml",
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

fiesta.sample(jax.random.PRNGKey(42))
fiesta.print_summary()
fiesta.save_results()
fiesta.plot_lightcurves()
fiesta.plot_corner()