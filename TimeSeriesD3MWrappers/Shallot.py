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

from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.2'
__contact__ = 'mailto:nklabs@newknowledge.com'


Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    num_shapelets = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.15, 
        upper_inclusive = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of shapelets, expressed as fraction of length of time series')
    min_shapelet_length = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.1, 
        upper_inclusive = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'base shapelet length, expressed as fraction of length of time series')
    num_shapelet_lengths = hyperparams.UniformInt(lower = 1, upper = 3, default = 2, 
        upper_inclusive = True, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    # default epoch size from https://tslearn.readthedocs.io/en/latest/auto_examples/plot_shapelets.html#sphx-glr-auto-examples-plot-shapelets-py
    epochs = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 200, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of training epochs')
    learning_rate = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.01, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    weight_regularizer = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.01, 
       upper_inclusive = True, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    long_format = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether the input dataset is already formatted in long format or not")
    pass


class Shallot(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies the shapelet classification algorithm to time series data. The shapelet 
        classification algorithm was introduced by Grabocka et al. in 
        https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf and learns discriminative subsequences 
        ("shapes") that can be used to classify series.
    
        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
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
                'version': '0.29.7',
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

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        
        self._params = {}
        self._X_train = None          # training inputs
        self._y_train = None          # training labels
        self._shapelets = Shapelets(self.hyperparams['epochs'], 
            self.hyperparams['min_shapelet_length'], self.hyperparams['num_shapelet_lengths'], 
            self.hyperparams['num_shapelets'], self.hyperparams['learning_rate'], self.hyperparams['weight_regularizer'],
            self.random_seed)  
        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self._hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits Shapelet classifier using training data from set_training_data and hyperparameters
        '''
        self._shapelets.fit(self._X_train, self._y_train)
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
        if not self.hyperparams['long_format']:
            inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
            ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
            inputs = d3m_DataFrame(ds2df_client.produce(inputs = inputs).value)

        # load and reshape training data
        # 'series_id' and 'value' should be set by metadata
        n_ts = len(inputs.d3mIndex.unique())
        ts_sz = int(inputs.shape[0] / n_ts)
        self._X_train = np.array(inputs.value).reshape(n_ts, ts_sz, 1)
        self._y_train = np.array(inputs.label.iloc[::ts_sz]).reshape(-1,)

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
        # temporary (until Uncharted adds conversion primitive to repo)
        if not self.hyperparams['long_format']:
            inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
            ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
            inputs = d3m_DataFrame(ds2df_client.produce(inputs = inputs).value)

        # parse values from output of time series formatter
        n_ts = len(inputs.d3mIndex.unique())
        ts_sz = int(inputs.shape[0] / n_ts)
        input_vals = np.array(inputs.value).reshape(n_ts, ts_sz, 1)

        # produce classifications using Shapelets
        classes = pandas.DataFrame(self._shapelets.predict(input_vals))
        output_df = pandas.concat([pandas.DataFrame(inputs.d3mIndex.unique()), classes], axis = 1)
        # get column names from metadata
        output_df.columns = ['d3mIndex', 'label']
        shallot_df = d3m_DataFrame(output_df)

        # first column ('d3mIndex')
        col_dict = dict(shallot_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        # confirm that this metadata still exists
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        shallot_df.metadata = shallot_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('predictions')
        col_dict = dict(shallot_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        shallot_df.metadata = shallot_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
        return CallResult(shallot_df)

if __name__ == '__main__':
        
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = Shallot.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'] 
    shallot_client = Shallot(hyperparams=hyperparams_class.defaults().replace({'shapelet_length': 0.4,'num_shapelet_lengths': 2, 'epochs':100}))
    shallot_client.set_training_data(inputs = input_dataset, outputs = None)
    shallot_client.fit()
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    results = shallot_client.produce(inputs = test_dataset)
    print(results.value)
