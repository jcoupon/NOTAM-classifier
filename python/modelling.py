# import all the necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class Modelling(object):
    """Class to perform modelling
    on NOTAM data.
    """

    def __init__(self):
        """Options
        """

        pass

    def read(self, path):
        """Read a csv file and 
        load it into Pandas data frame
        """

        # read file
        sys.stdout.write('Reading file...')
        self.__df = pd.read_csv(path, sep=',').set_index('item_id')      

        # save sample length
        self.N = len(self.__df)
        sys.stdout.write('done (found {} NOTAMs).\n'.format(self.N))

    def get_df(self):
        return self.__df

    def write(self, path):
        sys.stdout.write('Writting file...')
        self.__df.to_csv(path)
        sys.stdout.write('done.\n')
