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
import modelling

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

    path_in = args.path_in

    # output file path
    if args.path_out is None:
        # keep the path basename (without the extension)
        try:
            path_out, file_extension = os.path.splitext(path_in) #''.join(path_in.split('.')[:-1])
        except:
            # if the input file has no extension
            # keep it as is
            path_out = path_in
    
    if args.task == 'clean':
        clean(
            path_in, 
            path_out+'_clean.csv' if args.path_out is None else args.path_out,
            )

        return

    if args.task == 'train':

        if args.path_model is None:
            path_model = path_out+'_model_vectorize.pickle'
            path_model += ','+path_out+'_model_cluster.pickle'
        else:
            path_model = args.path_model            

        train(
            path_in,
            path_model,
            args.n_dim,
            n_samples_cluster=args.n_samples_cluster,
            vectorize_method=args.vectorize_method,
            )

        return

    raise Exception('task {} not recognized. Run main.py --help for details.'.format(args.task))


"""

-------------------------------------------------------------
Main functions
-------------------------------------------------------------


"""

def clean(path_in, path_out):
    """Read NOTAM csv file, perform cleaning
    and write into new csv file.
    """
    
    # create cleaner object and read the data
    cleaner = cleaning.Cleaning(path=path_in, sep=args.sep)

    # split the NOTAM into items (Q, A, B, C, etc.) 
    cleaner.split()

    # clean the unstructured (E) part
    cleaner.clean()

    # write result
    cleaner.write(path_out)

    return

def train(path_in, path_model, n_dim, vectorize_method='TFIDF-SVD', n_samples_cluster=None):
    """Read clean NOTAM csv file, train vectorize and 
    clustering (unsupervised) models and write model files.
    """

    # define the paths out for the models
    try:
        path_out_vectorize,path_out_cluster = path_model.split(',')
    except:
        raise Exception('train(): please provide 2 output paths separated by a coma (path_out_vectorize,path_out_cluster).')
    
    sys.stdout.write('Task: train.\n\nOutput model paths:\nvectorize:{0}\ncluster:{1}\n\n'.format(path_out_vectorize, path_out_cluster)); sys.stdout.flush()

    # create model training object
    model_train = modelling.ModelTraining(path_in)

    # vectorize the NOTAMs and do
    # dimensionality reduction
    model_train.vectorize(
        path_out=path_out_vectorize, 
        method=vectorize_method, 
        n_dim=n_dim,
        )

    # train and persist model
    if args.cluster_method == 'hierarch_cosine_average':
        method = 'hierarchical'
        method_options_dict = {'method': 'average', 'metric': 'cosine'}

    if args.cluster_method == 'hierarch_euclid_ward':
        method = 'hierarchical'
        method_options_dict = {'method': 'ward'}

    model_train.cluster_train(
        path_out=path_out_cluster, 
        method=method,
        method_options_dict=method_options_dict,
        n_samples=n_samples_cluster,
        )

    return

def predict(path_in, path_out):
    """Read clean NOTAM csv file, read model files and 
    clustering (unsupervised) models and write model files.
    """



    return


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
        'task',
        help='Task to perform among \'clean\',\'train\',\'test\' and \'predict\'',
    )

    parser.add_argument(
        'path_in', default=None,
        help='Input file path (csv file with NOTAMs)')

    parser.add_argument(
        '-path_out', default=None,
        help='Output file path. It will write a file with cleaned NOTAMs, \
cluster label and classification (group and importance) depending on the task. \
Default: input file path with task result appended to the name.')

    parser.add_argument(
        '-path_model', default=None,
        help='Output model file path (output for train and \
input for test and predict). Please provide 2 file names separated \
by a coma, first providing a path for the vectorizing model, \
then for the cluster model, e.g: model_vectorize.pickle,model_cluster.pickle')


    parser.add_argument('-seed', default=None, type=int, help='random seed')

    parser.add_argument('-sep', default=',', help='Separator for the input file')

    parser.add_argument('-n_samples_cluster', default=None, type=int, help='Number of samples for the training')

    parser.add_argument('-vectorize_method', default='TFIDF-SVD', help='Method to vectorize the NOTAMs. Default: TFIDF-SVD')

    parser.add_argument('-n_dim', type=int, default=50, help='Dimension of the vector. Default: 50')

    parser.add_argument('-cluster_method', default='hierarch_cosine_average', help='Metric to cluster the NOTAMs. Default: hierarch_cosine_average')


    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(seed=args.seed)

    main(args)
