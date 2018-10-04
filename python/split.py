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

from sklearn.model_selection import train_test_split


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
    df = pd.read_csv(args.input, sep=';').set_index('item_id')

    # split into training and test sample
    df_train, df_test = train_test_split(
        df, train_size=0.8, test_size=0.2, random_state=args.seed)
    
    # output paths
    path_train, path_test = args.output.split(',')

    df_train.to_csv(path_train)
    df_test.to_csv(path_test)

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
        help='Coma-separated output file paths for the training and test samples')

    parser.add_argument('-seed', default=20091982, type=int, help='random seed')

    args = parser.parse_args()

    main(args)
