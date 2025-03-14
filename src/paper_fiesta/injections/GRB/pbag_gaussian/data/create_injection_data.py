
from fiesta.inference.injection import InjectionPyblastafterglow
from fiesta.utils import write_event_data


param_dict = {"inclination_EM": 0.09,
              "log10_E0": 54.4, 
              "thetaCore": 0.16, 
              "alphaWing": 2.1, 
              "p": 2.5, 
              "log10_n0": -1.23, 
              "log10_epsilon_e": -2.06, 
              "log10_epsilon_B": -4.2, 
              "Gamma0": 444,
              "luminosity_distance": 40.0, 
              "redshift": 0., }


param_dict["trigger_time"] = 58849 # 01-01-2020 in mjd
FILTERS = ["radio-6GHz", "bessellv", "uvot::u", "X-ray-5keV"]

injection = InjectionPyblastafterglow(jet_type=0, 
                                      filters=FILTERS, 
                                      N_datapoints=75, 
                                      error_budget=0.2, 
                                      tmin=0.1, 
                                      tmax=2000, 
                                      trigger_time=param_dict["trigger_time"])
param_dict = injection.create_injection_from_file("../../../fiesta/flux_models/pyblastafterglow_gaussian/model/pyblastafterglow_raw_data.h5", param_dict)

write_event_data("./injection_pyblastafterglow_gaussian.dat", injection.data)
with open("param_dict.dat", "w") as o:
    o.write(str(param_dict))
