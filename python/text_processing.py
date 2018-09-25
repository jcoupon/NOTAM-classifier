# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import re
from functools import reduce

import pandas as pd


class SubstitutionRegex(object):
    def __init__(self, pattern, subs):
        self._regex = re.compile(pattern)
        self._subs = subs

    def __call__(self, text):
        return self._regex.sub(self._subs, text)


class SubstitutionString(object):
    def __init__(self, pattern, subs):
        self._pattern = pattern
        self._subs = subs

    def __call__(self, text):
        return text.replace(self._pattern, self._subs)


# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):
    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule

    num_subs = lambda m: '<number|{:02}>'.format(len(m.group(0)))

    substitutions = (
        # Numbers
        SubstitutionRegex(r'[0-9]+', num_subs),
        SubstitutionRegex(r'<number[|]\d{2}>[.]<number[|]\d{2}>', '<number|99>'),

        # Enumeration
        SubstitutionRegex(r'^[A-Z0-9][)]', ''),

        # Remove special characters
        SubstitutionString('(', ' '),
        SubstitutionString(')', ' '),
        SubstitutionString(',', ' '),
        SubstitutionString('.', ' '),
        SubstitutionString(':', ' '),
        SubstitutionString('?', ' '),
        SubstitutionString('+', ' '),
        SubstitutionString('=', ' '),
        SubstitutionString('"', ' '),
        SubstitutionString("'", ' '),

        # Handle synonyms
        SubstitutionRegex(r'\bACFT\b', 'AIRCRAFT'),
        SubstitutionRegex(r'\bACT\b', 'ACTIVE'),
        SubstitutionRegex(r'\bABV\b', 'ABOVE'),
        SubstitutionRegex(r'\bAD\b', 'AERODROME'),
        SubstitutionRegex(r'\bALT\b', 'ALTITUDE'),
        SubstitutionRegex(r'\bAMDT\b', 'AMENDMENT'),
        SubstitutionRegex(r'\bAP\b', 'AIRPORT'),
        SubstitutionRegex(r'\bAPCH\b', 'APPROACH'),
        SubstitutionRegex(r'\bAPN\b', 'APRON'),
        SubstitutionRegex(r'\bARR\b', 'ARRIVAL'),
        SubstitutionRegex(r'\bAVBL\b', 'AVAILABLE'),
        SubstitutionRegex(r'\bBLW\b', 'BELOW'),
        SubstitutionRegex(r'\bBTN\b', 'BETWEEN'),
        SubstitutionRegex(r'\bCAT\b', 'CATEGORY'),
        SubstitutionRegex(r'\bCTL\b', 'CONTROL'),
        SubstitutionRegex(r'\bCTN\b', 'CAUTION'),
        SubstitutionRegex(r'\bCLSD\b', 'CLOSED'),
        SubstitutionRegex(r'\bCOORD\b', 'COORDINATE'),
        SubstitutionRegex(r'\bCONST\b', 'CONSTRUCTION'),
        SubstitutionRegex(r'\bDEG\b', 'DEGREE'),
        SubstitutionRegex(r'\bDEP\b', 'DEPARTURE'),
        SubstitutionRegex(r'\bELEV\b', 'ELEVATION'),
        SubstitutionRegex(r'\bENR\b', 'ENROUTE'),
        SubstitutionRegex(r'\bEXC\b', 'EXCEPT'),
        SubstitutionRegex(r'\bFRNG\b', 'FIRING'),
        SubstitutionRegex(r'\bFLT\b', 'FLIGHT'),
        SubstitutionRegex(r'\bFLTCK\b', 'FLIGHTCHECK'),
        SubstitutionRegex(r'\bFLW\b', 'FOLLOW'),
        SubstitutionRegex(r'\bFREQ\b', 'FREQUENCY'),
        SubstitutionRegex(r'\bGND\b', 'GROUND'),
        SubstitutionRegex(r'\bGP\b', 'GLIDEPATH'),
        SubstitutionRegex(r'\bHGT\b', 'HEIGHT'),
        SubstitutionRegex(r'\bINTST\b', 'INTENSITY'),
        SubstitutionRegex(r'\bLGT\b', 'LIGHT'),
        SubstitutionRegex(r'\bLOC\b', 'LOCALIZER'),
        SubstitutionRegex(r'\bMAINT\b', 'MAINTENANCE'),
        SubstitutionRegex(r'\bMIL\b', 'MILITARY'),
        SubstitutionRegex(r'\bNR\b', 'NUMBER'),
        SubstitutionRegex(r'\bNA\b', 'UNSERVICEABLE'),
        SubstitutionRegex(r'\bN/A\b', 'UNSERVICEABLE'),
        SubstitutionRegex(r'\bOBST\b', 'OBSTACLE'),
        SubstitutionRegex(r'\bOPS\b', 'OPERATIONS'),
        #SubstitutionRegex(r'\bOPR\b', 'OPERATOR'),
        SubstitutionRegex(r'\bPROC\b', 'PROCEDURE'),
        SubstitutionRegex(r'\bPSN\b', 'POSITION'),
        SubstitutionRegex(r'\bRMK\b', 'REMARK'),
        SubstitutionRegex(r'\bRTE\b', 'ROUTE'),
        SubstitutionRegex(r'\bRWY\b', 'RUNWAY'),
        SubstitutionRegex(r'\bSFC\b', 'SURFACE'),
        SubstitutionRegex(r'\bSER\b', 'SERVICE'),
        SubstitutionRegex(r'\bSN\b', 'SNOW'),
        SubstitutionRegex(r'\bTAX\b', 'TAXI'),
        SubstitutionRegex(r'\bTFC\b', 'TRAFFIC'),
        SubstitutionRegex(r'\bTHR\b', 'THRESHOLD'),
        SubstitutionRegex(r'\bTWR\b', 'TOWER'),
        SubstitutionRegex(r'\bTWY\b', 'TAXIWAY'),
        SubstitutionRegex(r'\bU/S\b', 'UNSERVICEABLE'),
        SubstitutionRegex(r'\bVORTAC\b', 'VOR TACAN'),
        SubstitutionRegex(r'\bWI\b', 'WITHIN'),

        # Assign more meaning to numbers
        # Note '<' & '>' inverts the logic of \b to \B to detect
        # "non-alphanumeric" characters at the pattern boundary
        SubstitutionRegex(r'\B<number[|]\d{2}> ?(K|M)HZ\b', '<frequency>'),
        SubstitutionRegex(r'\bCH<number[|]02>[XY]\b', '<channel>'),
        SubstitutionRegex(r'\bFL<number[|]03>\B', '<flightlevel>'),
        SubstitutionRegex(r'\B<number[|]\d{2}> ?FT( ?(AMSL|AGL))?\b', '<height>'),
        SubstitutionRegex(r'\B<number[|]\d{2}>[NS] ?<number[|]\d{2}>[EW]\b', '<coordinate>'),
        SubstitutionRegex(r'\B<number[|]\d{2}> ?(M|KM|NM)\b', '<distance>'),
        SubstitutionRegex(r'\bRUNWAY ?<number[|]02>(R|L|C)?(/<number[|]02>(R|L|C)?)?(?!\w)', '<runway>'),

        # Remove length information from numbers again
        SubstitutionRegex(r'<number[|]\d{2}>', '<number>'),

        # Remove remaining special characters
        SubstitutionString('/', ' '),
        SubstitutionString('-', ' '),

        # Handle fixed phrases
        SubstitutionRegex(r'\bABOVE GROUND\b', 'AGL'),
        SubstitutionRegex(r'\bEN ROUTE\b', 'ENROUTE'),
        SubstitutionRegex(r'\bGLIDE PATH\b', 'GLIDEPATH'),
        SubstitutionRegex(r'\bFLIGHT CHECK\b', 'FLIGHTCHECK'),
        SubstitutionRegex(r'\bNOT AVAILABLE\b', 'UNSERVICEABLE'),
        SubstitutionRegex(r'\bOUT OF SERVICE\b', 'UNSERVICEABLE'),
        SubstitutionRegex(r'\bWORK IN PRORESS\b', 'WIP'),
    )


    dataframe1['text_azureml'] = dataframe1['text'].apply(
        lambda notam: reduce(lambda t, f: f(t), substitutions, notam.upper()))

    # Return value must be of a sequence of pandas.DataFrame
    return dataframe1,
