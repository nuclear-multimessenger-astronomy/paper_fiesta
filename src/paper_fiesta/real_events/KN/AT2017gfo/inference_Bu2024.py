import numpy as np

from fiesta.utils import load_event_data


########
# DATA #
########

data = load_event_data("../../data/AT2017gfo.dat")
trigger_time = 57982.52851852
FILTERS = data.keys()


for key in data.keys():
    print(np.min(data[key][:,0]-trigger_time))