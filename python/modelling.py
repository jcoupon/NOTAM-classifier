# import all the necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD


# python path where the script is located
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# default vocabulary path
PATH_VOCABULARY = DIR_PATH+'/vocabulary_dict.csv'

"""


Class to train the model


"""

class ModelTraining(object):

    def __init__(self, path=None):
        """Initialization
        """

        # load the dictionary in case
        # one is found. Note: will be
        # overwritten by build_vocabulary()
        try:
            self.__vocabulary_dict = read_vocabulary(
                path=PATH_VOCABULARY)
        except:
            self.__vocabulary_dict = None

        # default number of dimensions for the word vector
        self.__n_dim = 50

        # read the data if a path is given
        if path is not None:
            self.read(path)

    def read(self, path):
        """Read the clean NOTAM csv file and 
        load it into Pandas data frame
        """

        # read file
        sys.stdout.write('Reading file...')
        self.__df = pd.read_csv(path, sep=',').set_index('item_id')      

        # save sample length
        self.N = len(self.__df)
        sys.stdout.write('done (found {} NOTAMs).\n'.format(self.N))

    def build_vocabulary(self, path=PATH_VOCABULARY):
        """Build a vocabulary to be used for the 
        word count step. The input file is a csv file
        containing a column "text_clean" out of which 
        the vocabulary will be built.

        Write the vocabulary as a dictionary where 
        keys are terms and values are indices in 
        the feature matrix, or an iterable over terms.

        See http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        """

        vectorizer = CountVectorizer(stop_words='english')

        # get the NOTAMS as a corpus
        corpus = self.__df['text_clean'].fillna('').values

        # compute word_counts as a
        # sparse matrix
        sys.stdout.write('Building and saving the dictionary...')
        vectorizer.fit(corpus)

        # reset the indices so that there is 
        # no gap between 0 and the end
        vocabulary = {}
        for i,(k,v) in enumerate(vectorizer.vocabulary_.items()):
            vocabulary[k] = i

        # write dictionary
        pd.DataFrame\
            .from_dict(vocabulary,  orient="index")\
            .to_csv(path, header=False)
        sys.stdout.write('done.\n')

        # saves the dictionary into memory
        # Note that we could have passed 
        # vectorizer.vocabulary_ directly 
        # but invoking the read_vocabulary
        # function allows us to check it
        self.__vocabulary_dict = read_vocabulary(path=path)

        return

    def vectorize(self, random_state=None, n_dim=None):

        if n_dim is None:
            n_dim = self.__n_dim
        self.__vector = vectorize(
            self.__df, PATH_VOCABULARY=PATH_VOCABULARY, 
            n_dim=n_dim, random_state=None)

    def get_vector(self):
        return self.__vector

    def get_vocabulary_dict(self):
        return self.__vocabulary_dict
        
    def get_df(self):
        return self.__df


"""


Class to test the model or run the predictions


"""


class ModelPredict(object):
    """Class to perform modelling
    on NOTAM data.
    """

    def __init__(self):
        """Options
        """


        pass

    def read(self, path):
        """Read the clean NOTAM csv file and 
        load it into Pandas data frame
        """

        # read file
        sys.stdout.write('Reading file...')
        self.__df = pd.read_csv(path, sep=',').set_index('item_id')      

        # save sample length
        self.N = len(self.__df)
        sys.stdout.write('done (found {} NOTAMs).\n'.format(self.N))

    def cluster_train(self):

        # TODO
        #self.__model = modelling.cluster_train(
        #    X_train, model_options_dict=model_options_dict)


        pass

    def cluster_predict(self):



        pass






    def get_df(self):
        return self.__df

    def write(self, path):
        sys.stdout.write('Writting file...')
        self.__df.to_csv(path)
        sys.stdout.write('done.\n')


def cluster_train(
        X, n_samples=None, persist=None, 
        model_type='hierarchical', model_options_dict=None, 
        path_out=None):
    """Find clusters using scikit learn
    clustering algorithms
    """

    # limit the number of NOTAMs for the training
    if n_samples is not None:
        choice = np.random.randint(X.shape[0], size=n_samples)
    else:
        choice = range(len(X))

    #supress = df['supress'].values[choice]    
    if model_type == 'kmeans':
        if model_options_dict is None:
            # Default parameters
            model = KMeans(n_clusters=4, random_state=20091982)
        else:
            model = KMeans(**model_options_dict)
        model.fit(X[choice])

    if model_type == 'hierarchical':
        if model_options_dict is None:
            # Default parameters
            model = AgglomerativeClustering(n_clusters=7)
        else:
            model = AgglomerativeClustering(**model_options_dict)
        model.fit(X[choice])


    # persist model
    if path_out is not None:
        with open(path_out, 'wb') as file_out:
            pickle.dump(model, file_out)

    
    return model


def break_lines(input_text, stride=60):
    """Break lines for a list of texts"""

    output_text = []
    for text in input_text:
        string = ''
        try:
            if len(text) > stride:
                ids = range(0, len(text), stride)
                for start,end in zip(ids[:-1], ids[1:]):
                    string += text[start: end]+'<br>'
                string += text[end:]
            else:
                string = text
        except:
            pass
        
        output_text.append(string)
    
    return np.array(output_text)


def plot_clusters(
        X, labels, label_names=None, text=None, random_state=None, 
        html_out='clusters.html', interactive=True, do_break_lines=True,
        do_decompose=False):
    """Plot the clusters with their labels
    using plotly. If plotly is not available,
    fall back to non-interactive matplotlib plot
    """

    # import plotly
    try:
        import plotly
        import plotly.plotly as py
        import plotly.graph_objs as go
    except:
        sys.stdout.write('Plotly not found. \
Falling back to non-interactive plot (will write plot in graph.pdf).\n')
        interactive = False

    # label names
    label_types = np.unique(labels)
    if label_names is None:
        label_names = {l:str(l) for l in label_types}
    n_labels = len(label_types)

    if text is None:
        text = [str(l) for l in labels]
        do_break_lines = False

    # break lines of long texts
    if do_break_lines:
        text = break_lines(text)

    # Dimensionality reduction
    if do_decompose is True:
        sys.stdout.write('Performing dimensionality reduction...')
        decomposer = TruncatedSVD(n_components=2, random_state=random_state)
        X_decomposed = decomposer.fit_transform(X)
        sys.stdout.write('done\n')
    else:
        X_decomposed = X

    if interactive:
    
        plotly.offline.init_notebook_mode(connected=True)

        # loop over labels
        traces = []
        for i,n in enumerate(label_types):
            
            select = labels == n
            sys.stdout.write(
                'Plotting {0} points with label {1}\n'.format(
                    sum(select), label_names[n]))

            trace = go.Scatter(
                x = X_decomposed[:, 0][select],
                y = X_decomposed[:, 1][select],
                name = label_names[n],
                mode = 'markers',
                marker = dict(size = 5,),
                text = text[select],
                textposition='top left',

            )
            traces.append(trace)

        # main layout
        layout= go.Layout(
            autosize=False,
            width=800, height=800,
            title= '',
            hovermode= 'closest',
            xaxis= dict(
                title= 'x', ticklen= 5, 
                zeroline= False,gridwidth= 2,
            ),
            yaxis=dict(
                title= 'y', ticklen= 5,
                gridwidth= 2,
            ),
            showlegend= False
        )

        fig = dict(data=traces, layout=layout)
        plotly.offline.plot(fig, filename=html_out)

    else:
        
        import matplotlib.pyplot as plt

        #plt.style.use('seaborn-whitegrid')
        #plt.rc('pdf', fonttype=42)
        #sns.mpl.rc('figure', figsize = (10, 8))
        #sns.set_context('notebook', font_scale=2.0, rc={'lines.linewidth': 2.5})


        fig, ax = plt.subplots(figsize=(10, 10))
        for i,n in enumerate(label_types):
            select = labels == n
            sys.stdout.write('Plotting {0} points with label {1}\n'.format(sum(select), label_names[i]))
            ax.scatter(X_decomposed[:, 0][select], X_decomposed[:,1][select], alpha=1.0, label=label_names[i])
        ax.set_xlabel('Coord1')
        ax.set_ylabel('Coord2')    
        ax.legend(frameon=True)

        fig.savefig('graph.pdf')

    return


def read_vocabulary(path=PATH_VOCABULARY):
    """Read the vocabulary from path and return a dictionary.

    Function shared by all classes in this script.

    Return the vocabulary as a dictionary where 
    keys are terms and values are indices in 
    the feature matrix, or an iterable over terms.

    See http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    # the to_dict method returns a nested dictionary 
    # where the first layer is for the multiple columns (hence the [1]).
    # we force Pandas to keep "nan" as a string as it may 
    # exist in the NOTAMs
    dictionary = pd.read_csv(
        path, header=None, index_col=0, 
        keep_default_na=False, na_values=['']).to_dict()[1]
    N = len(list(dictionary.keys()))

    return dictionary


def vectorize(df, PATH_VOCABULARY=PATH_VOCABULARY, n_dim=50, random_state=None):
    """Vectorize the NOTAM and reduce the 
    dimensionality. First load a dictionary
    then count the words and run dimensionality
    reduction.  
    """

    # load the vocabulary
    vocabulary_dict = read_vocabulary(PATH_VOCABULARY)
    vectorizer = CountVectorizer(stop_words='english', vocabulary=vocabulary_dict)

    # get the NOTAMS as a corpus
    corpus = df['text_clean'].fillna('').values

    # compute word_counts as a
    # sparse matrix
    sys.stdout.write('Vectorizing the NOTAMs...')
    word_counts = vectorizer.transform(corpus)
    sys.stdout.write('done.\n')

    # vectorize word counts
    # reduce dimensionality using linear PCA
    # use TruncatedSVD which is really efficient
    # with sparse matrices
    sys.stdout.write('Performing dimensionality reduction...')
    decomposer = TruncatedSVD(n_components=n_dim, random_state=random_state)
    vector = decomposer.fit_transform(word_counts)
    sys.stdout.write('done.\n')

    return vector



def tsne(vector, n_dim=2, random_state=None):
    """Manifold t-SNE"""

    result = TSNE(
        n_components=2, verbose=2, random_state=random_state, perplexity=100)\
        .fit_transform(vector)

    return result


def get_cluster_purity(labels, classes):
    """Compute the purity of clusters
    given labels and classes.
    
    Loop over label types and compute 
    the purity of the class with the
    highest fraction.

    """

    # number of unique labels
    label_types = np.unique(labels)
    n_labels = len(label_types)

    # number of unique classes
    class_types = np.unique(classes)
    #n_c = len(class_types)

    # initialize results
    N = np.zeros(n_labels)
    purity = np.zeros(n_labels)

    # main loop
    for i,l in enumerate(label_types):

        select = labels == l
        N[i] = sum(select)

        fractions = []
        for j,c in enumerate(class_types):
            f = sum(classes[select] == c)/N[i]
            fractions.append(f)

        purity[i] = max(fractions)
        
        #print(N[i], '{0:.2f}'.format(purity[i]))


    return N, purity