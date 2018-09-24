# import all the necessary libraries
import os
import re
import pandas as pd
import numpy as np

class Cleaning(object):
    """Class to perform cleaning 
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
        self.__df = pd.read_csv(path, sep=';').set_index('item_id')
        
        # save sample length
        self.N = len(self.__df)

    def split(self):
        """Split the NOTAMs into different
        items
        """

        # unstructured text
        self.__df['text'] = self.__df['fulltext'].apply(get_text)

        # structured part
        # TODO

    def get_df(self):
        return self.__df


def get_text(NOTAM):
    """Extract unstructured 
    text "E)" from a NOTAM
    """

    # first check if the qualifier
    # is present. If not, do not 
    # return anything. These
    # NOTAMs will be addressed
    # differently (e.g. SNOWTAM)
    if not re.findall(r'Q\)', NOTAM):
        return ''

    # return the second item
    # among the E) to G) items
    try:
        # regex breakdown:
        # - E\)E\): sometimes "E)" is repeated twice
        # - \s[E-G]\) a space followed by a letter between
        # E and G, followed by ")"
        return re.split(r'E\)E\)|\s[E-G]\)', NOTAM)[1]
    except:
        return ''