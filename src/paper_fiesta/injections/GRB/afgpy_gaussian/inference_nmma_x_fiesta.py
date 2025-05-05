import os
import types

import numpy as np

import bilby
from bilby.gw.prior import Uniform, Constraint, PriorDict

from nmma.em.io import loadEvent
from nmma.em.model import GRBLightCurveModel
from nmma.em.em_likelihood import EMTransientLikelihood

from fiesta.inference.lightcurve_model import AfterglowFlux

########
# DATA #
########

data = loadEvent("./data/injection_afterglowpy_gaussian.dat")
trigger_time = 58849
FILTERS = list(data.keys())
FILTERS = [filt.replace("__", "::") for filt in FILTERS]

#########
# MODEL #
#########

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample, ["epsilon_tot", "thetaWing"]

model = AfterglowFlux(name="afgpy_gaussian_CVAE",
                      filters = FILTERS)

def generate_lightcurve(self, sample_times, parameters):
    t, mag = self.predict(parameters)
    
    for key in mag.keys():
        mag[key] = np.interp(sample_times, t, mag[key])

    return np.zeros(len(sample_times)), mag

model.generate_lightcurve = types.MethodType(generate_lightcurve, model)

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
sys_err = Uniform(minimum=0.3, maximum=1, name='sys_err')

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
              epsilon_tot = epsilon_tot,
              em_syserr = sys_err,
)

priors = PriorDict(dictionary = prior_dict, conversion_function = lambda x: conversion_function(x)[0])

likelihood_kwargs = dict(
    light_curve_model=model,
    light_curve_data=data,
    filters = FILTERS,
    trigger_time = trigger_time,
    tmin=1e-2,
    tmax = 200.0,
    priors=priors)

likelihood = EMTransientLikelihood(**likelihood_kwargs)

################
# LIKELIHOOD & #
# SAMPLING     #
################

sampler_kwargs = {}
outdir = f"./outdir_nmma_x_fiesta"
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