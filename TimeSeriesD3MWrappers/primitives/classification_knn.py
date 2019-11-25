import sys
import os
import numpy as np
import pandas as pd
import logging
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax

__author__ = 'Distil'
__version__ = '1.0.3'
__contact__ = 'mailto:jeffrey.gleason@yonder.co'

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    n_neighbors = hyperparams.UniformInt(
        lower = 0, 
        upper = sys.maxsize, 
        default = 5, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description='number of neighbors on which to make classification decision')
    distance_metric = hyperparams.Enumeration(
        default = 'euclidean', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        values = ['euclidean', 'dtw'],
        description = 'whether to use euclidean or dynamic time warping distance metric in KNN computation'
    )
    sample_weighting = hyperparams.Enumeration(
        default = 'uniform', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        values = ['uniform', 'inverse_distance'],
        description = 'whether to weight points uniformly or by the inverse of their distance'
    )

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
            {
                "type": "PIP",
                "package": "cython",
                "version": "0.29.7"
            },
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),)
            }
        ],
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

        self._knn = KNeighborsTimeSeriesClassifier(n_neighbors=self.hyperparams['n_neighbors'], 
            metric=self.hyperparams["distance_metric"],
            weights=self.hyperparams["sample_weighting"])
        self._scaler = TimeSeriesScalerMinMax()
        self._is_fit = False

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self._params = params

    def _get_cols(self, input_metadata):
        
        # find column with ts value through metadata
        grouping_column = input_metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/GroupingKey',))
        return grouping_column

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: time series data in long format (each row is one timestep of one series)

        outputs: vector / series / array containing 1 label / training time series
        '''

        # load and reshape training data
        self._output_columns = outputs.columns
        outputs = np.array(outputs)
        n_ts = outputs.shape[0]
        ts_sz = inputs.shape[0] // n_ts
        
        self._X_train = inputs.value.values.reshape(n_ts, ts_sz)
        self._y_train = np.array(outputs).reshape(-1,)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits KNN model using training data from set_training_data and hyperparameters
        """

        scaled = self._scaler.fit_transform(self._X_train)
        self._knn.fit(scaled, self._y_train)
        self._is_fit = True
        return CallResult(None, has_finished=self._is_fit)

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

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        # find column with ts value through metadata
        grouping_column = self._get_cols(inputs.metadata)

        n_ts = inputs.iloc[:, grouping_column[0]].nunique()
        ts_sz = inputs.shape[0] // n_ts
        x_vals = inputs.value.values.reshape(n_ts, ts_sz)

        # make predictions
        scaled = self._scaler.transform(x_vals)
        preds = self._knn.predict(scaled)

        # create output frame
        result_df = container.DataFrame({inputs.columns[self._output_columns[0]]: preds}, generate_metadata=True)
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 
            ('https://metadata.datadrivendiscovery.org/types/PredictedTarget'))

        return CallResult(result_df, has_finished=True)