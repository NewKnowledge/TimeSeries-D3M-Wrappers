import sys
import os.path
import numpy as np
import pandas as pd
import logging
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, params, base as metadata_base

from Sloth.classify import Knn

__author__ = 'Distil'
__version__ = '1.0.3'
__contact__ = 'mailto:jeffrey.gleason@yonder.co'

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    n_neighbors = hyperparams.UniformInt(lower = 0, upper = sys.maxsize, default = 5, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description='number of neighbors on which to make classification decision')

class Kanine(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies the k nearest neighbor classification algorithm to time series data. 
        The tslearn KNeighborsTimeSeriesClassifier implementation is wrapped.

        Training inputs: 1) Feature dataframe, 2) Label dataframe
        Outputs: Dataframe with predictions
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "2d6d3223-1b3c-49cc-9ddd-50f571818268",
        'version': __version__,
        'name': "kanine",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['time series', 'knn', 'k nearest neighbor', 'time series classification'],
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
            #  {
            #     'type': metadata_base.PrimitiveInstallationType.PIP,
            #     'package': 'cython',
            #     'version': '0.29.7',
            #  },
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

        self._knn = Knn(self.hyperparams['n_neighbors']) 

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits KNN model using training data from set_training_data and hyperparameters
        """

        self._knn.fit(self._X_train, self._y_train)
        return CallResult(None)
        
    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self._params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: time series data in long format (each row is one timestep of one series)

        outputs: vector / series / array containing 1 label / training time series
        '''

        # load and reshape training data
        outputs = np.array(outputs)
        n_ts = outputs.shape[0]
        ts_sz = inputs.shape[0] // n_ts

        # grab specific column b4 reshaping
        self._X_train = np.array(inputs.values).reshape(n_ts, ts_sz)
        self._y_train = np.array(outputs).reshape(-1,)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's classifications for new time series data

        Parameters
        ----------
        inputs : time series data in long format (each row is one timestep of one series)

        Returns
        ----------
        Outputs: The output is a dataframe with a column containing a predicted class for each input time series
        """

        # load and reshape test data
        grouping_column = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/GroupingKey')
        n_ts = len(inputs.iloc[:, grouping_column].unique())
        ts_sz = inputs.shape[0] // n_ts
        input_vals = np.array(inputs.value).reshape(n_ts, ts_sz)

        # make predictions
        preds = self._knn.predict(input_vals)

        # create output frame
        result_df = container.DataFrame({self.hyperparams['knn_predictions']: preds}, generate_metadata=True)
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 
            'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        return CallResult(result_df, has_finished=True)