#!/usr/bin/env python

"""
Jean coupon - 2018
scripts to split the NOTAM
into training and test sample
"""

import numpy as np
import os
import sys
import re
import pandas as pd


"""

-------------------------------------------------------------
global variables
-------------------------------------------------------------

"""


"""

-------------------------------------------------------------
main
-------------------------------------------------------------

"""


def main(args):
    """ Main function
    """


    # read input file
    df = pd.read_csv(args.input, sep=args.sep).set_index('item_id')

    n_in = len(df.index)

    # resample the data frame
    if args.n > n_in:
        raise Exception(
            'n ({0}) is larger than the number of rows ({1}). Choose a smaller number.'.format(args.n, n_in))
    
    choice = np.random.randint(n_in, size=args.n)

    # split into training and test sample
    df_resample = df.iloc[choice]
    
    df_resample.to_csv(args.output)

    return



"""

-------------------------------------------------------------
Main call and arguments
-------------------------------------------------------------


"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input',  default=None,
        help='input file path (csv file with NOTAMs)')

    parser.add_argument(
        'output',  default=None,
        help='Output file path')

    parser.add_argument(
        '-sep', default=',', help='Separator for the input file')

    parser.add_argument(
        '-n', default=1000, type=int, 
        help='Resampling number. Default: 1000.')

    parser.add_argument(
        '-seed', default=20091982, type=int, 
        help='Random seed')

    args = parser.parse_args()

    main(args)
