
from fiesta.inference.injection import InjectionKN


param_dict = {"inclination_EM": 0.4,
              "log10_mej_dyn": -2.5,
              "v_ej_dyn": 0.15,
              "Ye_dyn": 0.22,
              "log10_mej_wind": -1.2,
              "v_ej_wind": 0.08,
              "Ye_wind": 0.3,
              "luminosity_distance": 40.0, 
              "redshift": 0., }


param_dict["trigger_time"] = 58849 # 01-01-2020 in mjd
FILTERS = ["ps1::y", "besselli", "bessellv", "bessellux"]

injection = InjectionKN(filters=FILTERS, 
                        N_datapoints=75, 
                        error_budget=0.1, 
                        tmin=0.5, 
                        tmax=25,
                        trigger_time=param_dict["trigger_time"],
                        detection_limit=24
                        )

injection.create_injection(param_dict, "/home/aya/work/hkoehn/fiesta/fiesta/surrogates/KN/training_data/Bu2025_raw_data.h5")
injection.write_to_file("./injection_Bu2025.dat")
