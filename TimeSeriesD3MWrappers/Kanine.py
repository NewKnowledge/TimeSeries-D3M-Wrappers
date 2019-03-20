import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

from Sloth.classify import Knn

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

from timeseriesloader.timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.2'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    n_neighbors = hyperparams.UniformInt(lower = 0, upper = sys.maxsize, default = 5, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description='number of neighbors on which to make classification decision')
    pass

class Kanine(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    Produce primitive's classifications for new time series data. The input is a numpy ndarray of 
    size (number_of_time_series, time_series_length) containing new time series. 
    The output is a numpy ndarray containing a predicted class for each of the input time series.
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "2d6d3223-1b3c-49cc-9ddd-50f571818268",
        'version': __version__,
        'name': "kanine",
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
                'version': '0.28.5',
             },
             {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_classification.k_neighbors.Kanine',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.K_NEAREST_NEIGHBORS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._params = {}
        self._X_train = None        # training inputs
        self._y_train = None        # training outputs
        self._knn = Knn(self.hyperparams['n_neighbors']) 

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits KNN model using training data from set_training_data and hyperparameters
        """

        # fits ARIMA model using training data from set_training_data and hyperparameters
        self._knn.fit(self._X_train, self._y_train)
        return CallResult(None)
        
    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: numpy ndarray of size (number_of_time_series, time_series_length) containing training time series

        outputs: numpy ndarray of size (number_time_series,) containing classes of training time series
        '''

        # temporary (until Uncharted adds conversion primitive to repo)
        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})
        inputs = TimeSeriesFormatterPrimitive(hyperparams = hp).produce(inputs = inputs)

        # load and reshape training data
        inputs = inputs.value
        n_ts = len(inputs['0'].series_id.unique())
        ts_sz = int(inputs['0'].value.shape[0] / len(inputs['0'].series_id.unique()))
        self._X_train = np.array(inputs['0'].value).reshape(n_ts, ts_sz, 1)
        self._y_train = np.array(inputs['0'].label.iloc[::ts_sz]).reshape(-1,)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's classifications for new time series data

        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length) containing new time series 

        Returns
        ----------
        Outputs
            The output is a numpy ndarray containing a predicted class for each of the input time series
        """

        # temporary (until Uncharted adds conversion primitive to repo)
        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})
        inputs = TimeSeriesFormatterPrimitive(hyperparams = hp).produce(inputs = inputs)

        # parse values from output of time series formatter
        n_ts = len(inputs['0'].series_id.unique())
        ts_sz = int(inputs['0'].value.shape[0] / len(inputs['0'].series_id.unique()))
        inputs = inputs['0'].value.reshape(n_ts, ts_sz)

        classes = pandas.DataFrame(self._knn.predict(inputs))
        output_df = pandas.concat([pandas.DataFrame(inputs['0'].d3mIndex.unique()), classes], axis = 1)
        knn_df = d3m_DataFrame(output_df)

        # first column ('d3mIndex')
        col_dict = dict(knn_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        # confirm that this metadata still exists
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        knn_df.metadata = knn_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('predictions')
        col_dict = dict(knn_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        knn_df.metadata = knn_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
        return CallResult(knn_df)

if __name__ == '__main__':

    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = Kanine.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'] 
    kanine_client = Kanine(hyperparams=hyperparams_class.defaults().replace())
    kanine_client.set_training_data(inputs = input_dataset, outputs = None)
    kanine_client.fit()
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    results = kanine_client.produce(inputs = test_dataset)
    print(results.value)
