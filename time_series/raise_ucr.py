import pandas as pd
import numpy as np
from pyts.datasets import *

from utils import *


if __name__ == '__main__':

    names_list = ['Crop','FordB','FordA','NonInvasiveFetalECGThorax2','NonInvasiveFetalECGThorax1','PhalangesOutlinesCorrect','HandOutlines','TwoPatterns']
    coeffs = [0.1 , 0.2 ,0.3]

    for curr_dataset in names_list:
            (data_train, data_test, target_train, target_test)=fetch_ucr_dataset(curr_dataset, use_cache=True, data_home='/Users/aglenis/ucr_datasets/',
                                                                                 return_X_y=True)

            process_dataset(curr_dataset,data_train,coeffs)
