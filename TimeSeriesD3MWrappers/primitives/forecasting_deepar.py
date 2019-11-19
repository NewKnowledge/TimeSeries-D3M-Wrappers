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

from deepar.dataset.time_series import TimeSeries, TimeSeriesTest
from deepar.model.learner import DeepARLearner
import tensorflow as tf
import time

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:jeffrey.gleason@yonder.co'

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    emb_dim = hyperparams.UniformInt(
        lower = 8, 
        upper = 256, 
        default = 128, 
        upper_inclusive = True, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of cells to use in the categorical embedding component of the model')
    lstm_dim = hyperparams.UniformInt(
        lower = 8, 
        upper = 256, 
        default = 128, 
        upper_inclusive = True, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of cells to use in the lstm component of the model')
    epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize, 
        default = 100, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of training epochs')
    steps_per_epoch = hyperparams.UniformInt(
        lower = 1, 
        upper = 200, 
        default = 20, 
        upper_inclusive=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of steps to do per epoch')
    early_stopping_patience = hyperparams.UniformInt(
        lower = 0, 
        upper = sys.maxsize, 
        default = 5, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of epochs to wait before invoking early stopping criterion')
    early_stopping_delta = hyperparams.UniformInt(
        lower = 0, 
        upper = sys.maxsize, 
        default = 1, 
        upper_inclusive=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'early stopping will interpret change of < delta in desired direction \
            will increment early stopping counter state')
    learning_rate = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 1e-2, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'learning rate')
    batch_size = hyperparams.UniformInt(
        lower = 1, 
        upper = 256, 
        default = 16, 
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'batch size')
    dropout_rate = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 0.1, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'dropout to use in lstm model (input and recurrent transform)')
    window_size = hyperparams.UniformInt(
        lower = 10, 
        upper = sys.maxsize, 
        default = 20, 
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'window size of sampled time series in training process')
    negative_obs = hyperparams.UniformInt(
        lower = 0, 
        upper = 10, 
        default = 1, 
        upper_inclusive=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'whether to sample time series with padded observations before t=0 in training')
    val_split = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 0.2, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'proportion of training records to set aside for validation. Ignored \
            if iterations flag in `fit` method is not None')

class DeepAR(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies a deep autoregressive forecasting algorithm for time series
        prediction. The implementation is based off of this paper: https://arxiv.org/pdf/1704.04110.pdf
        and is implemented in AWS's Sagemaker interface.

        Training inputs: 1) Feature dataframe, 2) Target dataframe
        Outputs: Dataframe with predictions for specific time series at specific future time instances 
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "3410d709-0a13-4187-a1cb-159dd24b584b",
        'version': __version__,
        'name': "DeepAR",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['time series', 'forecasting', 'convolutional neural network', 'autoregressive'],
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
        # # a dependency which is not on PyPi.
        # 'installation': [
        #     {
        #         'type': metadata_base.PrimitiveInstallationType.PIP,
        #         'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
        #             git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #         ),
        #     }
        # ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_forecasting.convolutional_neural_net.DeepAR',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # set seed for reproducibility
        tf.random.set_seed(random_seed)

        self._is_fit = False
        self._new_train_data = False

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self._params = params

    def _drop_multiple_special_cols(self, col_list, col_type):
        if len(col_list) == 0:
            return None
        elif len(col_list) > 1:
            logger.warn(f'There are more than one {col_type} marked. This \
                primitive will use the first and drop other {col_type}s.')
            self._drop_cols += col_list[1:]
        return col_list[0]

    def _get_cols(self, input_metadata):
        
        self._drop_cols = []

        # get target idx (first column by default)
        target_columns = input_metadata.list_columns_with_semantic_types((
            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget', 
            'https://metadata.datadrivendiscovery.org/types/Target'))
        self._target_column = self._drop_multiple_special_cols(target_columns, 'target column')

        # get timestamp idx (first column by default)
        timestamp_columns = input_metadata.list_columns_with_semantic_types((
            "https://metadata.datadrivendiscovery.org/types/Time", 
            "http://schema.org/DateTime"))
        self._timestamp_column = self._drop_multiple_special_cols(timestamp_columns, 'timestamp column')

        # get grouping idx
        grouping_columns = input_metadata.list_columns_with_semantic_types(
            'https://metadata.datadrivendiscovery.org/types/GroupingKey')
        self._grouping_column = self._drop_multiple_special_cols(grouping_columns, 'grouping column')
        
        # get index_col (first index column by default)
        index_columns = input_metadata.list_columns_with_semantic_types(
            'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        self._index_column = self._drop_multiple_special_cols(index_columns, 'index column')

        # determine whether targets are count data 
        target_semantic_types = input_metadata.query_column_field(self._target_column, 'semantic_types')
        if "http://schema.org/Integer" in target_semantic_types:
            self._count_data = True
        elif "http://schema.org/Float" in target_semantic_types:
            self._count_data = False
        else:
            raise ValueError("Target column is not of type 'Integer' or 'Float'")

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: dataframe containing meta information about series and / or covariate (at each timestep)

        outputs: dataframe / series containing target observations at specific timesteps
        '''

        # combine inputs and outputs for internal TimeSeries object
        self._ts_frame = inputs.append_columns(outputs)

        # Parse cols needed for ts object
        self._get_cols(self._ts_frame.metadata)

        # drop cols if multiple grouping columns
        # logger.info(self._target_column)
        # logger.info(self._timestamp_column)
        # logger.info(self._grouping_column)
        # logger.info(self._index_column)
        # logger.info(self._count_data)
        logger.info(self._drop_cols)
        if len(self._drop_cols) > 0:
            self._ts_frame = self._ts_frame.remove_columns(self._drop_cols)

        # Create TimeSeries dataset objects 
        ts_object = TimeSeries(
            self._ts_frame,
            target_idx=self._target_column,
            timestamp_idx=self._timestamp_column,
            grouping_idx=self._grouping_column,
            index_col=self._index_column,
            count_data=self._count_data,
            negative_obs=self.hyperparams['negative_obs'],
            val_split=self.hyperparams['val_split']
        )

        # Create learner
        self._learner = DeepARLearner(
            ts_object,
            emb_dim=self.hyperparams['emb_dim'],
            lstm_dim=self.hyperparams['lstm_dim'],
            dropout=self.hyperparams['dropout_rate'],
            lr=self.hyperparams['learning_rate'],
            batch_size=self.hyperparams['batch_size'],
            train_window=self.hyperparams['window_size']
        )

        # save weights so we can start fitting from scratch (if desired by caller)
        self._learner.save_weights('model_initial_weights.h5')

        # mark that new training data has been set
        self._new_train_data = True

    def _create_data_and_model_no_val(self):

        # Create TimeSeries dataset object without validation set
        ts_object = TimeSeries(
            self._ts_frame,
            target_idx=self._target_column,
            timestamp_idx=self._timestamp_column,
            grouping_idx=self._grouping_column,
            index_col=self._index_column,
            count_data=self._count_data,
            negative_obs=self.hyperparams['negative_obs'],
            val_split=0
        )

        self._learner = DeepARLearner(
            ts_object,
            emb_dim=self.hyperparams['emb_dim'],
            lstm_dim=self.hyperparams['lstm_dim'],
            dropout=self.hyperparams['dropout_rate'],
            lr=self.hyperparams['learning_rate'],
            batch_size=self.hyperparams['batch_size'],
            train_window=self.hyperparams['window_size']
        )
        self._learner.save_weights('model_initial_weights.h5')

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits DeepAR model using training data from set_training_data and hyperparameters
        """

        # restore initial model weights if new training data
        if self._new_train_data:

            # only create new dataset object / model (w/out val) if new training data 
            if iterations is not None:
                self._create_data_and_model_no_val()
            self._learner.load_weights('model_initial_weights.h5')
        
        if iterations is None:
            iterations_set = False
            iterations = self.hyperparams['epochs']
        else:
            iterations_set = True

        # time training for 1 epoch so we can consider timeout argument thoughtfully
        validation = not iterations_set
        if timeout:
            logger.info('Timing the fitting procedure for one epoch so we \
                can consider timeout thoughtfully')
            start_time = time.time()
            self._learner.fit(validation=validation, 
                steps_per_epoch=self.hyperparams['steps_per_epoch'], 
                epochs=1, 
                stopping_patience=self.hyperparams['early_stopping_patience'], 
                stopping_delta=self.hyperparams['early_stopping_delta'], 
                tensorboard=False
            )
            epoch_time_estimate = time.time() - start_time
            # subract 1 for epoch that already happened and 1 more to be safe
            timeout_epochs = timeout // epoch_time_estimate - 2
            iters = min(timeout_epochs, iterations)
        else:
            iters = iterations

        # normal fitting
        logger.info(f'Fitting for {iters} iterations')
        start_time = time.time()

        _, iterations_completed = self._learner.fit(validation=validation, 
            steps_per_epoch=self.hyperparams['steps_per_epoch'], 
            epochs=iters, 
            stopping_patience=self.hyperparams['early_stopping_patience'], 
            stopping_delta=self.hyperparams['early_stopping_delta'], 
            tensorboard=False
        )
        logger.info(f'Fit for {iterations_completed} epochs, took {time.time() - start_time}s')

        # maintain primitive state (mark that training data has been used)
        self._new_train_data = False
        self._is_fit = True

        # use fitting history to set CallResult return values
        if iterations_set:
            has_finished = False
        elif iters < iteratins:
            has_finished = False
        else:
            has_finished = self._is_fit

        return CallResult(None, has_finished = has_finished, iterations_done = iterations_completed)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's predictions for specific time series at specific future time instances
            * these specific timesteps / series are specified implicitly by input dataset

        Parameters
        ----------
        inputs : dataframe containing meta information about series and / or covariate (at each timestep)

        Returns
        ----------
        Outputs: dataframe with predictions for specific time series at specific future time instances
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        # Create TimeSeriesTest object with saved metadata and train object

        # calculate max horizon

        # make predictions with learner

        # reshape (check input df from Grouping prim, we're looping through that column)

        # calculate desired prediction timesteps and slice preds

        # 1) TEST THROUGH HERE WITH DUMMY PREDS
        preds = [10] * inputs.shape[0]

        # create output frame
        result_df = container.DataFrame({inputs.columns[target_column[0]]: preds}, generate_metadata=True)
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 
            ('https://metadata.datadrivendiscovery.org/types/PredictedTarget'))

        return CallResult(result_df, has_finished=True)