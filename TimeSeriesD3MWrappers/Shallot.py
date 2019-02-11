import sys
import os.path
import numpy as np
import pandas
import typing
from json import JSONDecoder
from typing import List

from Sloth.classify import Shapelets

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

from .timeseries_loader import TimeSeriesLoaderPrimitive

__author__ = 'Distil'
__version__ = '1.0.1'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    shapelet_length = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.1, 
        upper_inclusive = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'base shapelet length, expressed as fraction of length of time series')
    num_shapelet_lengths = hyperparams.UniformInt(lower = 1, upper = 100, default = 2, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    # default epoch size from https://tslearn.readthedocs.io/en/latest/auto_examples/plot_shapelets.html#sphx-glr-auto-examples-plot-shapelets-py
    epochs = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 200, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of training epochs')
    learning_rate = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.01, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    weight_regularizer = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.01, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    pass


class Shallot(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    Produce primitive's classifications for new time series data The input is a numpy ndarray of 
    size (number_of_time_series, time_series_length, dimension) containing new time series. 
    The output is a numpy ndarray containing a predicted class for each of the input time series.
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d351fcf8-5d6c-48d4-8bf6-a56fe11e62d6",
        'version': __version__,
        'name': "shallot",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series', 'Shapelets'],
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
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),)
             }
         ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_classification.shapelet_learning.Shallot',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.STOCHASTIC_GRADIENT_DESCENT,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        
        self.volumes = volumes
        self._params = {}
        self._X_train = None          # training inputs
        self._y_train = None          # training outputs
        self._shapelets = None        # shapelet classifier

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits Shapelet classifier using training data from set_training_data and hyperparameters
        '''
        self._shapelets = Shapelets(self._X_train, self._y_train, self.hyperparams['epochs'], 
            self.hyperparams['shapelet_length'], self.hyperparams['num_shapelet_lengths'], 
            self.hyperparams['learning_rate'], self.hyperparams['weight_regularizer'])
        return CallResult(None)

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing training time series

        outputs: numpy ndarray of size (number_time_series,) containing classes of training time series
        '''
        # load and reshape training data
        ts_loader = TimeSeriesLoaderPrimitive(hyperparams = {"time_col_index":0, "value_col_index":1, "file_col_index":None})
        inputs = ts_loader.produce(inputs = inputs).value.values
        inputs = np.reshape(inputs, inputs.shape + (1,))
        self._X_train = inputs
        self._y_train = outputs['label']

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's classifications for new time series data

        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing new time series 

        Returns
        ----------
        Outputs
            The output is a numpy ndarray containing a predicted class for each of the input time series
        """
        # split filenames into d3mIndex (hacky)
        col_name = inputs.metadata.query_column(0)['name']
        d3mIndex_df = pandas.DataFrame([int(filename.split('_')[0]) for filename in inputs[col_name]])

        ts_loader = TimeSeriesLoaderPrimitive(hyperparams = {"time_col_index":0, "value_col_index":1, "file_col_index":None})
        inputs = ts_loader.produce(inputs = inputs).value.values
        inputs = np.reshape(inputs, inputs.shape + (1,))
        # add metadata to output
        # produce classifications using Shapelets
        classes = pandas.DataFrame(self._shapelets.PredictClasses(time_inputs))
        output_df = pandas.concat([d3mIndex_df, classes], axis = 1)
        shallot_df = d3m_DataFrame(output_df)

        # first column ('d3mIndex')
        col_dict = dict(shallot_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        shallot_df.metadata = shallot_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('predictions')
        col_dict = dict(shallot_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        shallot_df.metadata = shallot_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
        return CallResult(shallot_df)

if __name__ == '__main__':
        
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client_values = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"0"}))
    ds2df_client_labels = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
    df = d3m_DataFrame(ds2df_client_labels.produce(inputs = input_dataset).value)
    labels = d3m_DataFrame(ds2df_client_values.produce(inputs = input_dataset).value)  
    hyperparams_class = Shallot.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    shallot_client = Shallot(hyperparams=hyperparams_class.defaults().replace({'shapelet_length': 0.4,'num_shapelet_lengths': 2, 'epochs':100}))
    shallot_client.set_training_data(inputs = df, outputs = labels)
    shallot_client.fit()
    
    test_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    test_df = d3m_DataFrame(ds2df_client_values.produce(inputs = test_dataset).value)
    results = shallot_client.produce(inputs = test_df)
    print(results.value)
