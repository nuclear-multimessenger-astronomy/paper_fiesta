from fiesta.inference.injection import InjectionAfterglowpy


param_dict = {"inclination_EM": 0.05, 
              "log10_E0": 52.3, 
              "thetaCore": 0.03, 
              "alphaWing": 2.5, 
              "p": 2.12, 
              "log10_n0": -5., 
              "log10_epsilon_e": -1.2, 
              "log10_epsilon_B": -6.4, 
              "luminosity_distance": 40.0, 
              "redshift": 0., }


param_dict["trigger_time"] = 58849 # 01-01-2020 in mjd
FILTERS = ["radio-6GHz", "besselli", "bessellv", "uvot::b", "X-ray-5keV"]

injection = InjectionAfterglowpy(jet_type=0, 
                                 filters=FILTERS, 
                                 N_datapoints=75, 
                                 error_budget=0.1, 
                                 tmin=1e-2,
                                 tmax=200, 
                                 trigger_time=param_dict["trigger_time"])

injection.create_injection(param_dict)
injection.write_to_file("./injection_afterglowpy_gaussian.dat")
