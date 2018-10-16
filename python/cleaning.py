# import all the necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

INFO_NAMES = [
    'id', 'previous_id', 'type', 
    'new', 'NOTAM_conform', 'FIR', 'FIR_12', 'FIR_34',
    'code', 'code_23', 'code_45',
    'trafficind', 'purpose', 'scope', 'minflt', 'maxflt', 
    'lat', 'lng', 'location', 'n_locations',
    'radius', 'duration', 'end_est', 
    'diurnal_duration', 'perm',
    'len_txt', 'rvid',
    ]


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

        sys.stdout.write('Splitting items...'); sys.stdout.flush()

        # structured part
        NOTAMs = {c:[] for c in INFO_NAMES}

        #for i in range(10):
        for fulltext in self.__df['fulltext']:

            # catch error and display NOTAM accordingly
            try:
                NOTAM = get_unstructured(fulltext)
            except:
                print(fulltext)
                raise Exception(
                    'The above NOTAM\'s structured part failed to be converted')
            
            append_dict(NOTAMs, NOTAM)

        for key in NOTAMs.keys():
            self.__df[key] = NOTAMs[key]

        # unstructured part
        self.__df['text'] = self.__df['fulltext'].apply(get_text)
 
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



def append_dict(a, b):
    """ Append dictionary b to 
    dictionary a"""
    
    for k in a.keys():
        a[k].append(b[k])
    
    return

def str_dms_to_dd(coord_str):
    """ Convert sexadecimal to 
    decimal coordinates.
    
    https://www.latlong.net/lat-long-dms.html """
    
    try:
        d, m, s = coord_str[0:2], coord_str[2:4], 0.0
        lat = float(d) + float(m)/60 + float(s)/3600
        if coord_str[4] == 'S':
            lat = -lat
        d, m, s = coord_str[5:8], coord_str[8:10], 0.0
        lng = float(d) + float(m)/60 + float(s)/3600
        if coord_str[10] == 'W':
            lng = -lng
        radius = int(coord_str[11:])

    except:
        lat = np.nan
        lng = np.nan
        raise ValueError(
            'Unexpected string format for the coordinates {0}'.format(coord_str))
        
    return lat, lng, radius

def get_unstructured(txt):
    """ Extract NOTAM info 
    and return dictionary
    """
    
    #TODO: put rare codes and locations
    # into "rare encoders"
    
    # initialise output dictionary
    N = {c:None for c in INFO_NAMES}
    
    # N['txt'] = txt[txt.find('E)')+2: txt.find('F)')]
    # N['len_txt'] = len(N['txt'])
    # text length
    N['len_txt'] = len(txt[txt.find('E)')+2: txt.find('F)')])

    # and look over Q, A, B and C
    # then check each of the above is here    
    # high level split
    # find split position
    pos = {}
    for letter in ['Q', 'A', 'B', 'C', 'E', 'F', 'G']:
        pos[letter] = txt.find(letter+')')
        # exit if one of those is not found.
        # It means the NOTAM is unstructured 
        # and should be examined further
        if pos[letter] == -1 and letter in ['Q', 'A', 'B', 'C', 'E']:
            N['NOTAM_conform'] = 0
            return N
    
    # first part: NOTAM type and id
    words = re.findall(r'\S+', txt[0:pos['Q']])
    if len(words) == 2:
        N['id'], N['type'] = words
        N['previous_id'] = None
        N['new'] = 1
    elif len(words) == 3:
        # updated NOTAM
        N['id'], N['type'], N['previous_id'] = words
        N['new'] = 0
        
    # second part -> Q) NOTAM description
    words = txt[pos['Q']+2:pos['A']].split('/')
    # check that all 8 parts are here:
    if len(words) != 8:
        raise
        N['NOTAM_conform'] = False
        return N
    
    # contains 'XX' if multiple FIRs are involved
    N['FIR'] = words[0]
    N['FIR_12'] = words[0][0:2]
    N['FIR_34'] = words[0][2:4]

    # this is the core information of the NOTAM
    # split into 2 as well
    N['code'] = words[1]
    N['code_23'] = words[1][1:3]
    N['code_45'] = words[1][3:5]


    N['trafficind'] = words[2]
    N['purpose'] = words[3]
    N['scope'] = words[4]
    N['minflt'] = int(words[5])
    N['maxflt'] = int(words[6])
    N['lat'], N['lng'], N['radius'] = str_dms_to_dd(words[7])
    
    # second part -> A) location
    # ICAO indicator of the aerodrome or FIR
    # here we only record the first position
    # as well as the number of single FIR
    # the information on the geographical size 
    # of the NOTAM should alreayd be encapsulated in
    # the distance measurement and the first 
    # FIR information
    words = re.findall(r'\S+', txt[pos['A']+2:pos['B']])
    N['location'] = words[0]
    N['n_locations'] = len(words)
    
    # third and fourth part B) and C) -> time
    # here we record only the durantion length
    start_str = re.findall(r'\S+', txt[pos['B']+2:pos['C']])[0]
    
    # look if D) is present between C) and E)
    # only record whether it's set or not
    pos['D'] = txt.find('D)')
    if pos['D'] > -1 :
        N['diurnal_duration'] = True
        end_str = re.findall(r'\S+', txt[pos['C']+2:pos['D']])[0]
    else:
        N['diurnal_duration'] = False
        end_str = re.findall(r'\S+', txt[pos['C']+2:pos['E']])[0]

    # encode whether NOTAM has an estimated end
    # which requires a follow-up NOTAM afterwards
    if re.findall('EST', end_str):
        N['end_est'] = True
        end_str = end_str[:-3]
    else:
        N['end_est'] = False
    
    if end_str == 'PERM':
        N['perm'] = True
        N['duration'] = 0.0
    else:
        N['perm'] = False
        
        # time difference in decimal days
        start = datetime.strptime(start_str, "%y%m%d%H%M")
        end = datetime.strptime(end_str, "%y%m%d%H%M")
        N['duration'] = (end-start)/timedelta(days=1)
    
    # TODO: process F and G
    
    # if everything went well
    # set good to 1 and return
    # dictionary
    N['NOTAM_conform'] = True
    
    return N

