import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

from Sloth import Sloth

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

__author__ = 'Distil'
__version__ = '1.0.3'
__contact__ = 'mailto:jeffrey.gleason@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

# default values chosen for 56_sunspots 'sunspot.year' seed dataset
class Hyperparams(hyperparams.Hyperparams):
    index = hyperparams.UniformInt(lower = 0, upper = sys.maxsize, default = 2, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'], description='index of column to predict')
    n_periods = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 29, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'], description='number of periods to predict')
    seasonal = hyperparams.UniformBool(default = True, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="seasonal ARIMA prediction")
    seasonal_differencing = hyperparams.UniformInt(lower = 1, upper = 365, default = 12, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description='period of seasonal differencing')
    pass

class Parrot(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    Produce the primitive's prediction for future time series data. The output 
    is a list of length 'n_periods' that contains a prediction for each of 'n_periods' 
    future time periods. 'n_periods' is a hyperparameter that must be set before making the prediction.
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d473d487-2c32-49b2-98b5-a2b48571e07c",
        'version': __version__,
        'name': "parrot",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/parrot-d3m-wrapper",
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
                "type": "PIP",
                "package_uri": "git+https://github.com/NewKnowledge/sloth.git@82a1e08049531270256f38ca838e6cc7d1119223#egg=Sloth-2.0.3"
             },
             {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/parrot-d3m-wrapper.git@{git_commit}#egg=ParrotD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_forecasting.arima.Parrot',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.AUTOREGRESSIVE_INTEGRATED_MOVING_AVERAGE,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._params = {}
        self._X_train = None        # training inputs
        self._sloth = Sloth()       # Sloth model
        self._arima = None          # ARIMA classifier

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits ARIMA model using training data from set_training_data and hyperparameters
        """

        # fits ARIMA model using training data from set_training_data and hyperparameters
        self._arima = self._sloth.FitSeriesARIMA(self._X_train, 
                                                self.hyperparams['seasonal'],
                                                self.hyperparams['seasonal_differencing'])
        return CallResult(None)
        
    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Set primitive's training data

        Parameters
        ----------
        inputs : pandas data frame containing training data where first column contains dates and second column contains values
        
        """

        # use column according to hyperparameter index
        self._X_train = (inputs.iloc[:,self.hyperparams['index']].values).astype(np.float)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's prediction for future time series data

        Parameters
        ----------
        None

        Returns
        ----------
        Outputs
            The output is a data frame containing the d3m index and a forecast for each of the 'n_periods' future time periods
        """

        # add metadata to output
        # just take d3m index from input test set
        output_df = inputs['d3mIndex']
        # produce future foecast using arima
        future_forecast = pandas.DataFrame(self._sloth.PredictSeriesARIMA(self._arima, self.hyperparams['n_periods']))
        output_df = pandas.concat([output_df, future_forecast], axis=1)
        parrot_df = d3m_DataFrame(output_df)
        
        # first column ('d3mIndex')
        col_dict = dict(parrot_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        parrot_df.metadata = parrot_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('predictions')
        col_dict = dict(parrot_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = list(inputs)[self.hyperparams['index']]
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute',)
        parrot_df.metadata = parrot_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        return CallResult(parrot_df)

if __name__ == '__main__':

    # load data and preprocessing
    input_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/56_sunspots/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
    df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)
    hyperparams_class = Parrot.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    client = Parrot(hyperparams=hyperparams_class.defaults().replace({'index':2, 'n_periods':29, 'seasonal':True, 'seasonal_differencing':11}))
    client.set_training_data(inputs = df, outputs = None)
    client.fit()
    test_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/56_sunspots/TEST/dataset_TEST/datasetDoc.json')
    test_df = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value)
    results = client.produce(inputs = test_df)
    print(results.value)
