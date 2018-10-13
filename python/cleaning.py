# import all the necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Cleaning(object):
    """Class to perform cleaning 
    on NOTAM data.
    """

    def __init__(self, path=None, sep=','):
        """Options
        """

        self.__acronyms_dict = load_acronyms_dict()

        # read the data if a path is given
        if path is not None:
            self.read(path, sep=sep)

    def read(self, path, sep=','):
        """Read a csv file and
        load it into Pandas data frame
        """

        # read file
        sys.stdout.write('Reading file...'); sys.stdout.flush()
        self.__df = pd.read_csv(path, sep=sep).set_index('item_id')        

        # save sample length
        self.N = len(self.__df)
        sys.stdout.write('done (found {} NOTAMs).\n'.format(self.N)); sys.stdout.flush()

    def split(self):
        """Split the NOTAMs into 
        different items
        """

        # unstructured part
        sys.stdout.write('Splitting items...'); sys.stdout.flush()
        self.__df['text'] = self.__df['fulltext'].apply(get_text)
 
        # structured part
        # TODO
        sys.stdout.write('done.\n'); sys.stdout.flush()

    def clean(self):
        """Final step: clean the items 
        so they can be fed to ML algorithms.

        This method takes about 1mn on a 
        100k samples.

        TODO: implement progress bar:        
        from tqdm import tqdm
        tqdm.pandas()
        ...
        df.progress_apply()
        """

        # process unstructured part (replace numbers, etc.)
        sys.stdout.write('Cleaning unstructured part...'); sys.stdout.flush()
        self.__df['text_clean'] = self.__df['text']\
            .apply(lambda x:clean_unstructured(x, self.__acronyms_dict))
        sys.stdout.write('done.\n'); sys.stdout.flush()

        # add the "important" column
        # which is the inverse of the
        # "supress" column, if present
        try:
            self.__df['important'] = self.__df['supress'] == 0
        except:
            pass

        # structured part
        # TODO

    
    def get_df(self):
        return self.__df

    def write(self, path):
        sys.stdout.write('Writting file...'); sys.stdout.flush()
        self.__df.to_csv(path)
        sys.stdout.write('done.\n'); sys.stdout.flush()


def get_text(NOTAM):
    """Extract unstructured 
    text "E)" from a NOTAM
    """

    # first check whether the qualifier
    # is present. If not, do not 
    # return anything. These
    # NOTAMs will be addressed
    # differently (e.g. SNOWTAM)
    if not re.findall(r'\sQ\)', NOTAM):
        return ''

    # split items between E) to G)
    # return second item = E)
    try:
        # regex breakdown:
        # - E\)E\): matches "E)" repeated twice
        # - \s[E-G]\) matches a space followed by a letter between
        # E and G, followed by a ")"
        return re.split(r'E\)E\)|\s[E-G]\)', NOTAM)[1]
    except:
        return ''


def clean_unstructured(text, acronyms_dict):
    """Process and clean input text.

    Transform numbers and typical sentences
    into abbreviations/acronyms

    Here we assume that most abbreviations in aviation
    are not English stop words so that everything 
    can be transformed into lower case without diluting
    too much meaning.

    TODO check spelling?
    """

    # 1. transform integers and decimal numbers into <num>
    result = re.sub(r'\d+(\.\d+)*', '<num>', text)

    # 2. replace words/pattern by acronyms in dictionary
    for key in acronyms_dict.keys():
        result = re.sub(
            r'\b{}\b'.format(key), acronyms_dict[key], result.lower())

    # 3. transform coordinates into <coord>
    result = re.sub(r'<num>(n|s)<num>(e|w)', '<coord>', result)

    # 4. transform multiple spaces into one
    result = re.sub(r'\s+', ' ', result)

    # 5. replace anything which is non alpha-numeric
    # or a space or < or > by a space
    result = re.sub(r'[^a-z0-9\s<>]', ' ', result)

    # 6. return result
    return result #.upper()


def load_acronyms_dict(path=DIR_PATH+'/acronyms_dict.csv'):
    """Read path and return dictionary 
    in reverse order: 
    key,value -> dictionary[value] = key
    """
    
    dictionary = {}
    with open(path) as file_in:
        try:
            for line in file_in:
                key, value = line.replace('\n', '').split(',')
                dictionary[value.lower()] = key
        except:
            raise Exception(
                'Unexpected format for the \
input dictionary {}. It should be \"key,value\".'.format(path))

    return dictionary
