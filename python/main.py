#!/usr/bin/env python

"""
Jean coupon - 2018
scripts to clean NOTAMs, train, test 
and predict NOTAMS groups and importance
"""

import numpy as np
import os
import sys
import re
import pandas as pd

# add ./python to python path
#sys.path.insert(0, '../python')

# load local libraries
import cleaning
#import text_processing

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

    tasks = args.tasks.split(',')

    input_path = args.input

    # output file path
    if args.output is None:
        # keep the path basename (without the extension)
        try:
            output_path = ''.join(input_path.split('.')[:-1])
        except:
            # if the input file has no extension
            # keep it as is
            output_path = input_path

    if 'clean' in tasks:

        # create cleaner object
        cleaner = cleaning.Cleaning()

        # read the data
        cleaner.read(input_path)

        # split the NOTAM into items (Q, A, B, C, etc.) 
        cleaner.split()

        # clean the unstructured (E) part
        cleaner.clean()

        # write result
        cleaner.write(output_path+'_clean.csv')



    return


"""

-------------------------------------------------------------
Main functions
-------------------------------------------------------------


"""


"""

-------------------------------------------------------------
Utils
-------------------------------------------------------------


"""


"""

-------------------------------------------------------------
Main call and arguments
-------------------------------------------------------------


"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'tasks',
        default='clean,predict',
        help='Coma-separated tasks to perform among \'clean\',\'train\',\'test\' and \'predict\'. Default: clean,predict',
    )

    parser.add_argument(
        'input',  default=None,
        help='input file path (csv file with NOTAMs)')

    parser.add_argument(
        '-o', '--output', default=None,
        help='basename of the output file path. It will write a csv file with cleaned NOTAMs, features and classification (group and importance) at each stage of the process. Default: input file path with the task name appended to the name.')

    parser.add_argument('-seed', default=None, type=int, help='random seed')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(seed=args.seed)

    main(args)
