
from fiesta.inference.injection import InjectionAfterglowpy


param_dict = {"inclination_EM": 0.174, 
              "log10_E0": 54.4, 
              "thetaCore": 0.14, 
              "alphaWing": 3, 
              "p": 2.5, 
              "log10_n0": -1.23, 
              "log10_epsilon_e": -2.06, 
              "log10_epsilon_B": -4.2, 
              "luminosity_distance": 40.0, 
              "redshift": 0., }


param_dict["trigger_time"] = 58849 # 01-01-2020 in mjd
FILTERS = ["radio-6GHz", "bessellv", "uvot::u", "X-ray-5keV"]

injection = InjectionAfterglowpy(jet_type=0, 
                                 filters=FILTERS, 
                                 N_datapoints=75, 
                                 error_budget=0.2, 
                                 tmin=0.1, 
                                 tmax=2000, 
                                 trigger_time=param_dict["trigger_time"])
injection.create_injection(param_dict)
data = injection.data
injection.write_to_file("./injection_afterglowpy_gaussian.dat")
