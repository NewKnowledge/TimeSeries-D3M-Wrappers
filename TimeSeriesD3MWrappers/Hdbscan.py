import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

import hdbscan
from sklearn.cluster import DBSCAN
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame, dataframe_utils
from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.2'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.dataset.Dataset
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'HDBSCAN', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['DBSCAN', 'HDBSCAN'],
        description = 'type of clustering algorithm to use')
    eps = hyperparams.Uniform(lower=0, upper=sys.maxsize, default = 0.5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'maximum distance between two samples for them to be considered as in the same neigborhood, \
        used in DBSCAN algorithm')
    min_cluster_size = hyperparams.UniformInt(lower=2, upper=sys.maxsize, default = 5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'the minimum size of clusters')  
    min_samples = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default = 5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'The number of samples in a neighbourhood for a point to be considered a core point.')   
    long_format = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether the input dataset is already formatted in long format or not")
    pass

class Hdbscan(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    '''
        Primitive that applies Hierarchical Density-Based Clustering or Density-Based Clustering 
        algorithms to time series data. This is an unsupervised, clustering primitive, but has been
        representend as a supervised classification problem to produce a compliant primitive. 

        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "ca014488-6004-4b54-9403-5920fbe5a834",
        'version': __version__,
        'name': "hdbscan",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [
             {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'cython',
                'version': '0.29.7',
             },
             {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.clustering.hdbscan.Hdbscan',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DBSCAN,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self._hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})

        if self.hyperparams['algorithm'] == 'HDBSCAN':
            self.clf = hdbscan.HDBSCAN(min_cluster_size=self.hyperparams['min_cluster_size'],min_samples=self.hyperparams['min_samples'])
        else:
            self.clf = DBSCAN(eps=self.hyperparams['eps'],min_samples=self.hyperparams['min_samples'])
    
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length) containing new time series 

        Returns
        ----------
        Outputs
            The output is a dataframe containing a single column where each entry is the associated series' cluster number.
        """

        hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
        metadata_inputs = ds2df_client.produce(inputs = inputs).value
        
        # temporary (until Uncharted adds conversion primitive to repo)
        if not self.hyperparams['long_format']:
            formatted_inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            formatted_inputs = d3m_DataFrame(ds2df_client.produce(inputs = inputs).value)        

        # store information on target, index variable
        targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        target_names = [list(metadata_inputs)[t] for t in targets]
        index = metadata_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')

        # parse values from output of time series formatter
        n_ts = len(formatted_inputs.d3mIndex.unique())
        if n_ts == formatted_inputs.shape[0]:
            X_test = formatted_inputs.drop(columns = list(formatted_inputs)[index[0]])
            X_test = X_test.drop(columns = target_names).values
        else:
            ts_sz = int(formatted_inputs.shape[0] / n_ts)
            X_test = np.array(formatted_inputs.value).reshape(n_ts, ts_sz)

        # special semi-supervised case - during training, only produce rows with labels
        series = metadata_inputs[target_names] != ''
        if series.any().any():
            metadata_inputs = dataframe_utils.select_rows(metadata_inputs, np.flatnonzero(series))
            X_test = X_test[np.flatnonzero(series)]
        
        sloth_df = d3m_DataFrame(pandas.DataFrame(self.clf.fit_predict(X_test), columns=['cluster_labels']))
        # last column ('clusters')
        col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type(1)
        col_dict['name'] = 'cluster_labels'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        df_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
        df_dict_1 = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
        df_dict['dimension'] = df_dict_1
        df_dict_1['name'] = 'columns'
        df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
        df_dict_1['length'] = 1        
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
        
        return CallResult(utils_cp.append_columns(metadata_inputs, sloth_df))

if __name__ == '__main__':

    # Load data and preprocessing
    hyperparams_class = Hdbscan.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    hdbscan_client = Hdbscan(hyperparams=hyperparams_class.defaults())
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    results = hdbscan_client.produce(inputs = test_dataset)
    print(results.value)
