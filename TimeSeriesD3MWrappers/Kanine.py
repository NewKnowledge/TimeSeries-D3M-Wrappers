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

from timeseries_loader import TimeSeriesLoaderPrimitive

__author__ = 'Distil'
__version__ = '1.0.0'
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

        # load and reshape training data
        ts_loader = TimeSeriesLoaderPrimitive(hyperparams = {"time_col_index":0, "value_col_index":1, "file_col_index":None})
        inputs = ts_loader.produce(inputs = inputs).value.values
        self._X_train = inputs
        
        target = outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        self._y_train = outputs.iloc[:, target].values.reshape(-1,)

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

        # split filenames into d3mIndex (hacky)
        ## see if you can replace 'name' with metadata query for 'timeIndicator'
        col_name = inputs.metadata.query_column(0)['name']
        d3mIndex_df = pandas.DataFrame([int(filename.split('_')[0]) for filename in inputs[col_name]])

        ts_loader = TimeSeriesLoaderPrimitive(hyperparams = {"time_col_index":0, "value_col_index":1, "file_col_index":None})
        inputs = ts_loader.produce(inputs = inputs).value.values

        classes = pandas.DataFrame(self._knn.predict(inputs))
        output_df = pandas.concat([d3mIndex_df, classes], axis = 1)
        knn_df = d3m_DataFrame(output_df)

        # first column ('d3mIndex')
        col_dict = dict(knn_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        knn_df.metadata = knn_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('predictions')
        col_dict = dict(knn_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        knn_df.metadata = knn_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
        return CallResult(knn_df)

if __name__ == '__main__':

    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client_values = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"0"}))
    ds2df_client_labels = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
    df = d3m_DataFrame(ds2df_client_labels.produce(inputs = input_dataset).value)
    labels = d3m_DataFrame(ds2df_client_values.produce(inputs = input_dataset).value)  
    hyperparams_class = Kanine.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    kanine_client = Kanine(hyperparams=hyperparams_class.defaults())
    kanine_client.set_training_data(inputs = df, outputs = labels)
    kanine_client.fit()
    
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    test_df = d3m_DataFrame(ds2df_client_values.produce(inputs = test_dataset).value)
    results = kanine_client.produce(inputs = test_df)
    print(results.value)
