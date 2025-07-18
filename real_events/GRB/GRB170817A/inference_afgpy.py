import os

import numpy as np
import jax
import jax.numpy as jnp

from fiesta.inference.prior import Uniform, Constraint
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.utils import load_event_data




########
# DATA #
########

data = load_event_data("../../data/GRB170817A.dat")
trigger_time = 57982.52851852
FILTERS = data.keys()

#########
# MODEL #
#########

model = AfterglowFlux(name="afgpy_gaussian_CVAE",
                      filters = FILTERS)


#########
# PRIOR #
#########

inclination_EM = Uniform(xmin=0.0, xmax=np.pi/4, naming=['inclination_EM'])
log10_E0 = Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0'])
thetaCore = Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore'])
alphaWing = Uniform(xmin = 0.2, xmax = 3.5, naming= ["alphaWing"])
thetaWing = Constraint(xmin = 0, xmax = np.pi/2, naming = ["thetaWing"])
log10_n0 = Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0'])
p = Uniform(xmin=2.01, xmax=3.0, naming=['p'])
log10_epsilon_e = Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e'])
log10_epsilon_B = Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B'])
epsilon_tot = Constraint(xmin = 0., xmax = 1., naming=["epsilon_tot"])

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
              epsilon_tot]

prior = ConstrainedPrior(prior_list, conversion_function)

################
# LIKELIHOOD & #
# SAMPLING     #
################
  
  
detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          FILTERS,
                          tmin=1.,
                          tmax=2000.0,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 43.58, "redshift": 0.009727}
                          )




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
                outdir = outdir)

fiesta.sample(jax.random.PRNGKey(42))#, initial_guess=jnp.array([[0.36262065, 53.03262   ,  0.05980359,  3.4285772 , -3.0439792 , 2.1210535 , -2.313587  , -2.2309036 ,  0.30090034]]))
fiesta.print_summary()
fiesta.save_results()
fiesta.plot_lightcurves()
fiesta.plot_corner()
