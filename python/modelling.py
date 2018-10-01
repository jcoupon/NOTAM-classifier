# import all the necessary libraries
import os
import sys
import re
import pandas as pd
import numpy as np
import pickle

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE



dir_path = os.path.dirname(os.path.realpath(__file__))

class Modelling(object):
    """Class to perform modelling
    on NOTAM data.
    """

    def __init__(self):
        """Options
        """


        pass

    def read(self, path):
        """Read a csv file and 
        load it into Pandas data frame
        """

        # read file
        sys.stdout.write('Reading file...')
        self.__df = pd.read_csv(path, sep=',').set_index('item_id')      

        # save sample length
        self.N = len(self.__df)
        sys.stdout.write('done (found {} NOTAMs).\n'.format(self.N))

    def cluster_train(self):



        pass

    def cluster_predict(self):



        pass






    def get_df(self):
        return self.__df

    def write(self, path):
        sys.stdout.write('Writting file...')
        self.__df.to_csv(path)
        sys.stdout.write('done.\n')

def break_lines(input_text, stride=60):
    """Break lines for a list of texts"""

    output_text = []
    for text in input_text:
        string = ''
        ids = range(0, len(text), stride)
        for start,end in zip(ids[:-1], ids[1:]):
            string += text[start: end]+'<br>'
        string += text[end:]

        output_text.append(string)
    
    return np.array(output_text)


def plot_clusters(
        X, labels, label_names=None, text=None, random_state=None, 
        html_out='clusters.html', interactive=True, do_break_lines=True):
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
        label_names = [str(l) for l in label_types]
    n_labels = len(label_types)

    if text is None:
        text = labels

    # break lines of long texts
    if do_break_lines:
        text = break_lines(text)

    # Dimensionality reduction
    decomposer = TruncatedSVD(n_components=2, random_state=None)

    sys.stdout.write('Performing dimensionality reduction...')
    X_decomposed = decomposer.fit_transform(X)
    sys.stdout.write('done\n')

    if interactive:
    
        plotly.offline.init_notebook_mode(connected=True)

        # loop over labels
        traces = []
        for i,n in enumerate(label_types):
            
            select = labels == n    
            sys.stdout.write(
                'Plotting {0} points with label {1}\n'.format(
                    sum(select), label_names[i]))

            trace = go.Scatter(
                x = X_decomposed[:, 0][select],
                y = X_decomposed[:, 1][select],
                name = label_names[i],
                mode = 'markers',
                marker = dict(size = 5,),
                text = text[select]
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
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.legend(frameon=True)

        fig.savefig('graph.pdf')


    


    return