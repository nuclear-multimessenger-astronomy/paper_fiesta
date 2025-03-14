import os

import numpy as np

import bilby
from bilby.gw.prior import Uniform, Constraint, PriorDict

from nmma.em.io import loadEvent
from nmma.em.model import GRBLightCurveModel
from nmma.em.em_likelihood import EMTransientLikelihood

########
# DATA #
########

data = loadEvent("./data/injection_afterglowpy_gaussian.dat")
trigger_time = 58849
FILTERS = data.keys()

#########
# MODEL #
#########

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample, ["epsilon_tot", "thetaWing"]

model = GRBLightCurveModel(parameter_conversion = conversion_function,
                           jet_type = 0,
                           filters = FILTERS)

#########
# PRIOR #
#########

inclination_EM = Uniform(minimum=0.0, maximum=np.pi/2, name='inclination_EM')
log10_E0 = Uniform(minimum=47.0, maximum=57.0, name='log10_E0')
alphaWing = Uniform(minimum=0.2, maximum=3.4, name = 'alphaWing')
thetaCore = Uniform(minimum=0.01, maximum=np.pi/5, name='thetaCore')
log10_n0 = Uniform(minimum=-6.0, maximum=2.0, name='log10_n0')
p = Uniform(minimum=2.01, maximum=3.0, name='p')
log10_epsilon_e = Uniform(minimum=-4.0, maximum=0.0, name='log10_epsilon_e')
log10_epsilon_B = Uniform(minimum=-8.0, maximum=0.0, name='log10_epsilon_B')
thetaWing = Constraint(minimum=0, maximum=np.pi/2, name='thetaWing')
epsilon_tot = Constraint(minimum=0, maximum=1, name='epsilon_tot')

luminosity_distance = 40.0
redshift = 0.
timeshift = 0.

prior_dict = dict(inclination_EM = inclination_EM, 
              log10_E0 = log10_E0, 
              thetaCore = thetaCore, 
              alphaWing = alphaWing,
              log10_n0 = log10_n0, 
              p = p, 
              log10_epsilon_e = log10_epsilon_e, 
              log10_epsilon_B = log10_epsilon_B,
              luminosity_distance = luminosity_distance, 
              redshift = redshift,
              timeshift = timeshift,
              thetaWing = thetaWing,
              epsilon_tot = epsilon_tot
)

priors = PriorDict(dictionary = prior_dict, conversion_function = lambda x: conversion_function(x)[0])

likelihood_kwargs = dict(
    light_curve_model=model,
    light_curve_data=data,
    filters = FILTERS,
    trigger_time = trigger_time,
    tmax = 2000.0,
    error_budget = 0.5,
    param_conv_func = conversion_function,
    priors=priors)

likelihood = EMTransientLikelihood(**likelihood_kwargs)

################
# LIKELIHOOD & #
# SAMPLING     #
################

sampler_kwargs = {}
outdir = f"./nmma"
if not os.path.exists(outdir):
    os.makedirs(outdir)

result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="pymultinest",
    outdir=outdir,
    label=f'injection_gaussian',
    nlive=1024,
    seed=42,
    check_point_delta_t=3600,
    **sampler_kwargs,
)

result.save_posterior_samples()