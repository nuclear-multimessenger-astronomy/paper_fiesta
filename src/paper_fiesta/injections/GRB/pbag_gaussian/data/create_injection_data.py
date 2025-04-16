
from fiesta.inference.injection import InjectionPyblastafterglow


param_dict = {"inclination_EM": 0.09,
              "log10_E0": 52.3, 
              "thetaCore": 0.02, 
              "alphaWing": 2.1, 
              "p": 2.5, 
              "log10_n0": -0.23, 
              "log10_epsilon_e": -1.56, 
              "log10_epsilon_B": -3.2, 
              "Gamma0": 800,
              "luminosity_distance": 40.0, 
              "redshift": 0., }


param_dict["trigger_time"] = 58849 # 01-01-2020 in mjd
FILTERS = ["radio-6GHz", "besselli", "bessellv", "uvot::b", "X-ray-5keV"]

injection = InjectionPyblastafterglow(jet_type=0, 
                                      filters=FILTERS, 
                                      N_datapoints=75, 
                                      error_budget=0.1, 
                                      tmin=1e-2, 
                                      tmax=200, 
                                      trigger_time=param_dict["trigger_time"])

injection.create_injection(param_dict, "/work/koehn1/fiesta/fiesta/surrogates/GRB/training_data/pyblastafterglow_gaussian_raw_data.h5")
injection.write_to_file("./injection_pyblastafterglow_gaussian.dat")
