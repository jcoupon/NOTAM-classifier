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
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.neighbors import KNeighborsClassifier

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from gensim import corpora, models, similarities
from collections import defaultdict

# python path where the script is located
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# default vocabulary path
PATH_VOCABULARY = DIR_PATH+'/vocabulary_dict.csv'

# output file for the cluster model
PATH_VECTORIZE_MODEL = DIR_PATH+'/vectorize_model.pickle'

# random seed
RANDOM_STATE = None

# output file for the cluster model
PATH_CLUSTER_MODEL = DIR_PATH+'/cluster_model'

# default number of dimension for the word vector
N_DIM = 50

"""


Class to train the model


"""

class ModelTraining(object):

    def __init__(self, path=None, n_dim=N_DIM):
        """Initialization
        """

        self.__vocabulary_dict = None

        # default number of dimensions for the word vector
        self.__n_dim = n_dim

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
            self, path_out=PATH_VECTORIZE_MODEL, 
            method='TFIDF-SVD', 
            path_vocabulary=PATH_VOCABULARY, 
            random_state=None, n_dim=None):
        """Vectorize the NOTAMs and get the dictionary """

        if n_dim is None:
            n_dim = self.__n_dim
        self.__vector = vectorize(
            self.__df, path=path_out, path_vocabulary=path_vocabulary, 
            n_dim=n_dim, method=method, random_state=None)


        #self.__vocabulary_dict = read_vocabulary(path_vocabulary)

        return

    def cluster_train(
            self, path_out=PATH_CLUSTER_MODEL,
            method='hierarchical',
            method_options_dict = {'method': 'ward'},
            random_state=RANDOM_STATE, n_samples=None):
        """ Train clusters with hierarchical clustering 
        and persist model as the vector plus labels
        that will be used with k-NN for testing and
        predicting"""

        # run clustering
        sys.stdout.write('Training clusters (method:{0}, options:{1})...'.format(
            method, method_options_dict)); sys.stdout.flush()
        
        model = find_clusters_train(
            self.__vector, method=method,
            method_options_dict=method_options_dict,
            path_out=path_out, n_samples=n_samples, 
            random_state=random_state)
        sys.stdout.write('done.\n'); sys.stdout.flush()

#    def cluster_predict(
#            self, path_in=PATH_CLUSTER_MODEL, 
#            method='hierarchical'):
#        # run clustering
#        sys.stdout.write('Predicting (clusters)...'); sys.stdout.flush()
#        self.__labels = find_clusters_predict(
#            self.__vector, path_in, method=method,
#            n_samples=None)
#        self.__df['cluster_labels'] = self.__labels
#        sys.stdout.write('done.\n'); sys.stdout.flush()

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

    def __init__(self, path=None, n_dim=N_DIM):
        """Initialization
        """

        # default number of dimensions for the word vector
        self.__n_dim = n_dim

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
            self, path_in=PATH_VECTORIZE_MODEL, 
            method='TFIDF-SVD', 
            path_vocabulary=PATH_VOCABULARY,
            do_build_vocabulary=False, 
            
            random_state=None):
        """Vectorize the NOTAMs"""

        self.__vector = vectorize(
            self.__df, path=path_in, 
            path_vocabulary=path_vocabulary, 
            do_train=False, method=method,
        )

        return

    def cluster_predict(
            self, path_in=PATH_CLUSTER_MODEL, 
            method='hierarchical', dist=None):

        # run clustering
        sys.stdout.write('Predicting clusters...'); sys.stdout.flush()
        self.__cluster_labels = find_clusters_predict(
            self.__vector, path_in, method=method,
            n_samples=None, dist=dist)
        self.__df['cluster_labels'] = self.__cluster_labels
        sys.stdout.write('done.\n'); sys.stdout.flush()

    def cluster_test(
        self, path_in=PATH_CLUSTER_MODEL, class_name='important'):

        sys.stdout.write('Testing clusters...\n'); sys.stdout.flush()

        with open(path_in, 'rb') as file_in:
            _, _, Z = pickle.load(file_in)

        n_clusters_list = []
        f_pure_list = []
        dist_list = []

        # choose min distance so that it covers 
        # a wide range of the number of clusters 
        log_d_min = np.log10(np.quantile(Z[:,2], 0.95))
        log_d_max = np.log10(max(Z[:,2]))

        for dist in np.logspace(log_d_min, log_d_max, 30):

            n_clusters = sum(Z[:,2] > dist)
            
            labels = find_clusters_predict(
                self.__vector, path_in, dist=dist)

            N, purity, f_pure = get_cluster_purity(labels, self.__df[class_name])        
            
            f_pure_list.append(f_pure)
            n_clusters_list.append(n_clusters)
            dist_list.append(dist)

            sys.stdout.write('n_clusters={0}, dist={1:.2f}, f_pure={2:.2f}\n'.format(
                n_clusters, dist, f_pure)); sys.stdout.flush()

        sys.stdout.write('done.\n'); sys.stdout.flush()

        return n_clusters_list, dist_list, f_pure_list


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
        X, n_samples=None,
        method='hierarchical', method_options_dict=None, 
        path_out=None, random_state=None):
    """Find clusters using scikit learn
    clustering algorithms.

    Return the labels
    """

    # limit the number of NOTAMs for the training
    if n_samples is not None:
        choice = np.random.randint(X.shape[0], size=n_samples)
    else:
        choice = range(len(X))

    if method == 'hierarchical':

        Z = linkage(X[choice], **method_options_dict)

        d = np.quantile(Z[:,2], 0.995)
        labels = fcluster(Z, d, criterion='distance')

        # persist vector, labels and linkage matrix
        if path_out is not None:
            with open(path_out, 'wb') as file_out:
                pickle.dump((X[choice], labels, Z), file_out)
        
        return labels, Z

    if method == 'kmeans':
        if method_options_dict is None:
            # Default parameters
            model = KMeans(n_clusters=4, random_state=random_state)
        else:
            model = KMeans(**method_options_dict)
        model.fit(X[choice])

        labels = model.labels_

        # persist model
        if path_out is not None:
            with open(path_out, 'wb') as file_out:
                pickle.dump(model, file_out)

        return labels

    if method == 'hierarchical_scikit_learn':

        if method_options_dict is None:
            # Default parameters
            model = AgglomerativeClustering(n_clusters=7)
        else:
            model = AgglomerativeClustering(**method_options_dict)
        model.fit(X[choice])

        labels = model.labels_

        # persist model
        if path_out is not None:
            with open(path_out, 'wb') as file_out:
                pickle.dump((X[choice], model.labels_), file_out)
    
        return labels

    raise Exception('find_clusters_train(): method {} not recognized'.format(method))

def find_clusters_predict(
        X, path_in, n_samples=None,
        method='hierarchical', dist=None,
        method_options_dict=None, ):
    """Find clusters using scikit learn
    clustering or nearest neighbour algorithms.

    dist is used only with method='hierarchical'
    and allows to define the cluster distance 
    to re-compute the trained clusters from the 
    linkage matrix (Z) 
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

        return labels

    if method == 'hierarchical': 

        # read model
        with open(path_in, 'rb') as file_in:
            X_train, labels_train, Z = pickle.load(file_in)

        if dist is not None:
            labels_train = fcluster(Z, dist, criterion='distance')

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train, labels_train)
        labels = model.predict(X[choice])

        return labels

    raise Exception('find_clusters_predict(): method {} not recognized'.format(method))


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


def build_vocabulary(
        corpus, path=PATH_VOCABULARY, method='scikit-learn', min_freq=0):
    """Build a vocabulary to be used for the 
    word count step. The input file is a csv file
    containing a column "text_clean" out of which 
    the vocabulary will be built.

    Write the vocabulary as a dictionary where 
    keys are terms and values are indices in 
    the feature matrix, or an iterable over terms.

    See http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    if method == 'scikit-learn':
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

    if method == 'gensim':

        # remove common words and tokenize
        # all punctuation and special characters
        # must have been removed during cleaning
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist]
                for document in corpus]

        # replace words that appear only rarely
        if min_freq > 0:
            frequency = defaultdict(int)
            for text in texts:
                for token in text:
                    frequency[token] += 1
            
            texts = [['rare_token' if frequency[token] <= min_freq else token for token in text]
                    for text in texts]

        vocabulary = corpora.Dictionary(texts)
        vocabulary.save_as_text(PATH_VOCABULARY)

        return vocabulary

    raise Exception('build_vocabulary(): method {} not recognized'.format(method))


def read_vocabulary(path=PATH_VOCABULARY, method='scikit-learn'):
    """Read the vocabulary from path and return a dictionary.

    Function shared by all classes in this script.

    Return the vocabulary as a dictionary where 
    keys are terms and values are indices in 
    the feature matrix, or an iterable over terms.

    See http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    if method == 'scikit-learn':

        # the to_dict method returns a nested dictionary 
        # where the first layer is for the multiple columns (hence the [1]).
        # we force Pandas to keep "nan" as a string as it may 
        # exist in the NOTAMs
        vocabulary = pd.read_csv(
            path, header=None, index_col=0, 
            keep_default_na=False, na_values=['']).to_dict()[1]
        
        # N = len(list(dictionary.keys()))

        return vocabulary

    if method == 'gensim':

        vocabulary = corpora.Dictionary.load_from_text(PATH_VOCABULARY)

        return vocabulary


    raise Exception('read_vocabulary(): method {} not recognized'.format(method))

def vectorize(
        df, path = PATH_VECTORIZE_MODEL,
        path_vocabulary=PATH_VOCABULARY, do_train=True,
        method='BOW-SVD', text_col_name='text_clean', n_dim=50, 
        random_state=None):
    """Vectorize the NOTAM and reduce the
    dimensionality. First load a dictionary
    then count the words and run dimensionality
    reduction.

    Methods:
    - TFIDF-SVD
    - TFIDF-NMF
    - BOW-SVD
    - BOW-LDA
    - word2vec

    See http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    and http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/

    """

    # get the NOTAMS as a corpus
    # replace empty NOTAMs by 'empty'
    # so that their cosine distance is
    # not 0
    corpus = df[text_col_name].fillna('empty_NOTAM').values


    if method in ['TFIDF-SVD', 'TFIDF-NMF', 'BOW-SVD', 'BOW-LDA']:

        if do_train:

            sys.stdout.write('Vectorizing NOTAMs (method:{0}, n_dim:{1})...'.format(method, n_dim)); sys.stdout.flush()

            # build the vocabulary
            vocabulary_dict = build_vocabulary(
                corpus, path=path_vocabulary, method='scikit-learn')

            # compute word_counts as a sparse matrix
            if method.split('-')[0] == 'BOW':
                vectorizer = CountVectorizer(
                    stop_words='english', vocabulary=vocabulary_dict)
            if method.split('-')[0] == 'TFIDF':
                vectorizer = TfidfVectorizer(
                    stop_words='english', vocabulary=vocabulary_dict)

            word_counts = vectorizer.fit_transform(corpus)

            # vectorize word counts
            # reduce dimensionality using linear PCA-like
            # technique called TruncatedSVD 
            # -> efficient with sparse matrices
            if method.split('-')[1] == 'SVD':
                decomposer = TruncatedSVD(n_components=n_dim, random_state=random_state)
            if method.split('-')[1] == 'LDA':
                decomposer = LatentDirichletAllocation(n_components=n_dim, random_state=random_state)
            if method.split('-')[1] == 'NMF':
                decomposer = NMF(n_components=n_dim, random_state=random_state)
    
            vector = decomposer.fit_transform(word_counts)

            # persist models
            if path is not None:
                with open(path, 'wb') as file_out:
                    pickle.dump((vectorizer, decomposer), file_out)

        else:
            # can load the dictionary if necessary
            #vocabulary_dict = read_vocabulary(
            #    path_vocabulary, method='scikit-learn')

            if path is not None:
                with open(path, 'rb') as file_in:
                    vectorizer, decomposer = pickle.load(file_in)            
            else:
                raise Exception(
                    'vectorize(): a model file must be provided when predicting (do_train = False).')
        
            sys.stdout.write('Vectorizing NOTAMs...'); sys.stdout.flush()

            word_counts = vectorizer.transform(corpus)
            vector = decomposer.transform(word_counts)

        sys.stdout.write('done.\n'); sys.stdout.flush()
        return vector

    if method in ['word2vec']:

        # see https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df
        # and https://medium.com/@sherryqixuan/topic-modeling-and-pyldavis-visualization-86a543e21f58
        if do_train:

            sys.stdout.write('Vectorizing NOTAMs (method:{0}, n_dim:{1})...'.format(method, n_dim)); sys.stdout.flush()

            # build the vocabulary
            vocabulary_dict = build_vocabulary(
                corpus, path=path_vocabulary, method='gensim')
            
            texts = [text.lower().split() for text in corpus]
            word_counts = [vocabulary_dict.doc2bow(text) for text in texts]

            if method == 'word2vec':
                model = models.Word2Vec(texts, min_count=1, size=n_dim) #, id2word=vocabulary_dict, num_topics=n_dim)
                vectors = combine_word_vectors(model, texts)
        
                # persist model
                if path is not None:
                    with open(path, 'wb') as file_out:
                        pickle.dump(model, file_out)

                sys.stdout.write('done.\n')
                return vectors

            if method == 'LDA':

                model = models.LdaModel(
                    word_counts, id2word=vocabulary_dict, num_topics=n_dim)

                # TODO
                n_samples = len(word_counts)
                vectors = np.zeros((n_samples, n_dim))
                for i,wc in enumerate(word_counts):
                    v = np.array(model[wc]) 
                    vectors[i, v[:, 0].astype(int)] = v[:, 1]  
                
                sys.stdout.write('done.\n')
                return vectors

        else:
            # load the dictionary
            vocabulary_dict = read_vocabulary(
                path_vocabulary, method='gensim')

            if path is not None:
                with open(path, 'rb') as file_in:
                    model = pickle.load(file_in)            
            else:
                raise Exception(
                    'vectorize(): a model file must be provided when predicting (do_train = False).')

            texts = [text.lower().split() for text in corpus]
            word_counts = [vocabulary_dict.doc2bow(text) for text in texts]

            if method == 'word2vec':
                vectors = combine_word_vectors(model, texts)

                sys.stdout.write('done.\n')
                return vectors


        sys.stdout.write('done.\n')
        return None

    raise Exception('vectorize(): method {} not recognized'.format(method))


def combine_word_vectors(model, texts):
    """Combine word vectors over a NOTAM
    using the average."""

    vectors = np.zeros((len(texts), model.vector_size))
    vector = np.zeros(model.vector_size)

    for i,c in enumerate(texts):
        vector = 0
        n_words = 0
        for word in c:
            try:
                vector += model.wv[word]
                n_words += 1
            except:
                pass
        if n_words > 0:
            vectors[i, :] = vector/n_words

    return vectors



def tsne(vector, n_dim=2, random_state=None, perplexity=100):
    """Manifold t-SNE"""

    result = TSNE(
        n_components=2, verbose=2, random_state=random_state, perplexity=perplexity)\
        .fit_transform(vector)

    return result


def get_cluster_purity(labels, classes, min_purity=0.8):
    """Compute the purity of clusters
    given labels and classes.
    
    Loop over label types and compute 
    the purity of the class with the
    highest fraction.

    """

    # check that the number of 
    # unique classes is less than 2
    classes_types = np.unique(classes)
    if len(classes_types) > 2:
        raise Exception('get_cluster_purity(): the number of classes should not exceed 2.')

    # define the purity function 
    def purity(row):
        """Compute the ratio between class count and 
        total count. Then pick the class with the highest
        fraction."""
        ratio = row[('class', 'sum')]/row[('class', 'count')] 
        return max(1.0-ratio, ratio)    

    # compute the class ratio after grouping by cluster label
    df = pd.DataFrame.from_dict({'label':labels, 'class':classes})
    df_counts = df.groupby(by='label').agg({'class': ['sum', 'count']})

    # purity and number per cluster
    purity = df_counts.apply(purity, axis=1).values
    N = df_counts[('class', 'count')].values

    # fraction of pure cluster weighted by 
    # NOTAM counts, i.e. fraction of 
    # purely classified NOTAMs (=precision)
    f_pure = sum(N[purity > min_purity])/sum(N)

    return N, purity, f_pure
