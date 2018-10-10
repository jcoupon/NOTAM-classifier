# import all the necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier




# python path where the script is located
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# default vocabulary path
PATH_VOCABULARY = DIR_PATH+'/vocabulary_dict.csv'

# random seed
RANDOM_STATE = None

CLUSTER_MODEL_PATH = DIR_PATH+'cluster_model'

"""


Class to train the model


"""

class ModelTraining(object):

    def __init__(self, path=None):
        """Initialization
        """

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


    def vectorize(
            self, method='BOW', path_vocabulary=PATH_VOCABULARY, 
            random_state=None, n_dim=None):
        """Vectorize the NOTAMs and get the dictionary """

        if n_dim is None:
            n_dim = self.__n_dim
        self.__vector = vectorize(
            self.__df, path_vocabulary=path_vocabulary, 
            n_dim=n_dim, method=method, random_state=None)

        self.__vocabulary_dict = read_vocabulary(path_vocabulary)

        return


    def cluster_train(
            self, path_out=CLUSTER_MODEL_PATH,
            method='hierarchical', n_clusters=50, 
            random_state=RANDOM_STATE, n_samples=None):
        """ Train clusters with hierarchical clustering 
        and persist model as the vector plus labels
        that will be used with k-NN for testing and
        predicting"""

        # run clustering
        sys.stdout.write('Training (clusters)...'); sys.stdout.flush()
        method_options_dict = {'n_clusters': n_clusters}
        model = find_clusters_train(
            self.__vector, method=method,
            method_options_dict=method_options_dict,
            path_out=path_out, n_samples=n_samples, 
            random_state=random_state)
        sys.stdout.write('done.\n'); sys.stdout.flush()

    def cluster_predict(
            self, path_in=CLUSTER_MODEL_PATH, 
            method='hierarchical'):
        # run clustering
        sys.stdout.write('Predicting (clusters)...'); sys.stdout.flush()
        self.__labels = find_clusters_predict(
            self.__vector, path_in, method=method,
            n_samples=None)
        self.__df['cluster_labels'] = self.__labels
        sys.stdout.write('done.\n'); sys.stdout.flush()

    def get_vector(self):
        return self.__vector

    def get_cluster_labels(self):
        return self.__labels

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

    def __init__(self, path=None):
        """Initialization
        """

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

    def vectorize(
            self, method='BOW', path_vocabulary=PATH_VOCABULARY,
            do_build_vocabulary=False, 
            random_state=None, n_dim=None):
        """Vectorize the NOTAMs"""

        if n_dim is None:
            n_dim = self.__n_dim
        self.__vector = vectorize(
            self.__df, path_vocabulary=path_vocabulary, 
            n_dim=n_dim, method=method, random_state=None)

        return

    def cluster_predict(
            self, path_in=CLUSTER_MODEL_PATH, 
            method='hierarchical'):
        # run clustering
        sys.stdout.write('Predicting (clusters)...'); sys.stdout.flush()
        self.__cluster_labels = find_clusters_predict(
            self.__vector, path_in, method=method,
            n_samples=None)
        self.__df['cluster_labels'] = self.__cluster_labels
        sys.stdout.write('done.\n'); sys.stdout.flush()


    def visualize(
            self, method='t-SNE', random_state=None, 
            path_out='clusters_TSNE.csv'):
        """Compute the visualization tools.

        default:
        t-SNE representation
        additional info can be
        found here https://distill.pub/2016/misread-tsne/


        TODO: call plot_cluster
        """

        if method == 't-SNE':

            # the perplexity drives the number of 
            # neighbours in t-SNE. A higher number 
            # tend to make results look smoother
            perplexity = 100

            # compute the t-SNE representation
            # it takes about 30 mn for a sample
            # of 20,000
            vector_TSNE = tsne(self.__vector, random_state=random_state, perplexity=perplexity)

            # persist the t-SNE coordinates
            pd.DataFrame(vector_TSNE).to_csv(path_out)

            #path_test_TSNE = path_test.replace('.csv', '_TSNE_perp{}.csv'.format(perplexity))

            # the result to a file
            #if False:
            #else:
            #    test_TSNE = pd.read_csv(path_test_TSNE)[['0', '1']].values


            return


        raise Exception('visualize(): method {} not recognized'.format(method))

    def get_vector(self):
        return self.__vector

    def get_df(self):
        return self.__df

    def get_cluster_labels(self):
        return self.__cluster_labels

    def write(self, path):
        sys.stdout.write('Writting file...')
        self.__df.to_csv(path)
        sys.stdout.write('done.\n')


def find_clusters_train(
        X, n_samples=None, persist=None, 
        method='hierarchical', method_options_dict=None, 
        path_out=None, random_state=None):
    """Find clusters using scikit learn
    clustering algorithms
    """

    # limit the number of NOTAMs for the training
    if n_samples is not None:
        choice = np.random.randint(X.shape[0], size=n_samples)
    else:
        choice = range(len(X))

    if method == 'kmeans':
        if method_options_dict is None:
            # Default parameters
            model = KMeans(n_clusters=4, random_state=random_state)
        else:
            model = KMeans(**method_options_dict)
        model.fit(X[choice])

        # persist model
        if path_out is not None:
            with open(path_out, 'wb') as file_out:
                pickle.dump(model, file_out)

    if method == 'hierarchical':
        if method_options_dict is None:
            # Default parameters
            model = AgglomerativeClustering(n_clusters=7)
        else:
            model = AgglomerativeClustering(**method_options_dict)
        model.fit(X[choice])

        # persist model
        if path_out is not None:
            with open(path_out, 'wb') as file_out:
                pickle.dump((X[choice], model.labels_), file_out)
    
    return model


def find_clusters_predict(
        X, path_in, n_samples=None, persist=None, 
        method='hierarchical', method_options_dict=None):
    """Find clusters using scikit learn
    clustering algorithms
    """

    # limit the number of NOTAMs for the test
    if n_samples is not None:
        n_samples = min(n_samples, len(X))
        choice = np.random.randint(X.shape[0], size=n_samples)
    else:
        choice = range(len(X))

    if method == 'kmeans':
        
        # read model
        with open(path_in, 'rb') as file_in:
            model = pickle.load(file_in)

        # evaluate labels
        labels = model.predict(X[choice])

    if method == 'hierarchical': 

        # read model
        with open(path_in, 'rb') as file_in:
            X_train, labels_train = pickle.load(file_in)

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train, labels_train)
        labels = model.predict(X[choice])

    return labels


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
        do_decompose=False, emphasize_label=None):
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

            size = 5
            if emphasize_label is not None and emphasize_label != n:
                size = 1
            
            if X_decomposed.shape[1] > 2:
                trace = go.Scatter3d(
                    x = X_decomposed[:, 0][select],
                    y = X_decomposed[:, 1][select],
                    z = X_decomposed[:, 2][select],
                    name = label_names[n],
                    mode = 'markers',
                    marker = dict(size = size,),
                    text = text[select],
                    textposition='top left',

                )
            else:
                trace = go.Scatter(
                    x = X_decomposed[:, 0][select],
                    y = X_decomposed[:, 1][select],
                    name = label_names[n],
                    mode = 'markers',
                    marker = dict(size = size,),
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


def build_vocabulary(corpus, path=PATH_VOCABULARY):
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

    # compute word_counts as a
    # sparse matrix
    vectorizer.fit(corpus)

    # reset the indices so that there is 
    # no gap between 0 and the end
    vocabulary = {}
    for i,(k,v) in enumerate(vectorizer.vocabulary_.items()):
        vocabulary[k] = i

    # write dictionary
    if path is not None:
        pd.DataFrame\
            .from_dict(vocabulary,  orient="index")\
            .to_csv(path, header=False)

    return vocabulary


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


def vectorize(
        df, path_vocabulary=PATH_VOCABULARY, do_build_vocabulary=True, 
        method='BOW', text_col_name='text_clean', n_dim=50, 
        random_state=None):
    """Vectorize the NOTAM and reduce the 
    dimensionality. First load a dictionary
    then count the words and run dimensionality
    reduction.  
    """

    # get the NOTAMS as a corpus
    corpus = df[text_col_name].fillna('').values

    if method == 'BOW':
        # compute word_counts as a
        # sparse matrix
        sys.stdout.write('Vectorizing the NOTAMs...')

        # build or load the vocabulary
        # usually the dictionary is built during 
        # the training and saved
        if do_build_vocabulary:
            vocabulary_dict = build_vocabulary(corpus, path=path_vocabulary)
        else:
            vocabulary_dict = read_vocabulary(path_vocabulary)

        vectorizer = CountVectorizer(stop_words='english', vocabulary=vocabulary_dict)
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

    if method == 'TF-IDF':
        # compute word_counts as a
        # sparse matrix
        sys.stdout.write('Vectorizing the NOTAMs...')
        vectorizer = TfidfVectorizer(stop_words='english')
        word_counts = vectorizer.fit_transform(corpus)
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

    if method == 'word2vec':

        # remove common words and tokenize
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist]
                for document in corpus]
        # remove words that appear only once
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [[token for token in text if frequency[token] > 1] for text in texts]


        return texts

    raise Exception('vectorize: method {} not recognized'.format(method))


def tsne(vector, n_dim=2, random_state=None, perplexity=100):
    """Manifold t-SNE"""

    result = TSNE(
        n_components=2, verbose=2, random_state=random_state, perplexity=perplexity)\
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

    # fraction of pure cluster weighted by 
    # NOTAM counts, i.e. fraction of 
    # purely classified NOTAMs (=precision)
    f_pure = sum(N[purity > 0.8])/sum(N)

    return N, purity, f_pure



