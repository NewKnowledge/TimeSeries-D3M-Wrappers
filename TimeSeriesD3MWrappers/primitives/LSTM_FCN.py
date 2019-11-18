import sys
import os
import numpy as np
import pandas
import time
import logging

from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.exceptions import PrimitiveNotFittedError

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from TimeSeriesD3MWrappers.models.model_utils import generate_lstmfcn, LSTMSequence, LSTMSequenceTest
from sklearn.preprocessing import LabelEncoder

__author__ = 'Distil'
__version__ = '1.0.2'
__contact__ = 'mailto:jeffrey.gleason@yonder.co'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    attention_lstm = hyperparams.UniformBool(
        default = False, 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="whether to use attention in the lstm component of the model")
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
        default = 2000, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of training epochs')
    learning_rate = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 1e-3, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'learning rate')
    batch_size = hyperparams.UniformInt(
        lower = 1, 
        upper = 512, 
        default = 32, 
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'batch size')
    dropout_rate = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 0.2, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'dropout rate (before lstm layer in model)')
    val_split = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 0.2, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'proportion of training records to set aside for validation. Ignored \
            if iterations flag in `fit` method is not None')
    early_stopping_patience = hyperparams.UniformInt(
        lower = 0, 
        upper = sys.maxsize, 
        default = 100, 
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of epochs to wait before invoking early stopping criterion')
    use_multiprocessing = hyperparams.UniformBool(
        default = True, 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="whether to use multiprocessing in training")
    num_workers = hyperparams.UniformInt(
        lower = 1, 
        upper = 16, 
        default = 8, 
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of workers to do if using multiprocessing threading')

class LSTM_FCN(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies a LSTM FCN (LSTM fully convolutional network) for time
        series classification. The implementation is based off this paper: 
        https://ieeexplore.ieee.org/document/8141873 and this base library: 
        https://github.com/NewKnowledge/LSTM-FCN.
    
        Training inputs: 1) Feature dataframe, 2) Label dataframe
        Outputs: Dataframe with predictions
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "a55cef3a-a7a9-411e-9dde-5c935ff3504b",
        'version': __version__,
        'name': "lstm_fcn",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series', 'convolutional neural network', 'cnn', 'lstm', 'time series classification'],
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
        #  'installation': [
        #      {
        #         'type': metadata_base.PrimitiveInstallationType.PIP,
        #         'package': 'cython',
        #         'version': '0.29.7',
        #      },
        #      {
        #         'type': metadata_base.PrimitiveInstallationType.PIP,
        #         'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
        #             git_commit=utils.current_git_commit(os.path.dirname(__file__)),)
        #      }
        #  ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # set seed for reproducibility
        tf.random.set_seed(random_seed)

        self._is_fit = False
        self._new_train_data = False

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def _get_cols(self, input_metadata):
        
        # find column with ts value through metadata
        attribute_cols = input_metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Attribute',))
        grouping_column = input_metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/GroupingKey',))
        target_column = input_metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget', 
            'https://metadata.datadrivendiscovery.org/types/Target'))
        value_column = list(set(attribute_cols) - set(grouping_column) - set(target_column))
        return value_column, target_column, grouping_column

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

        # find column with ts value through metadata
        value_column, _, _ = self._get_cols(inputs.metadata)

        self._X_train = inputs.iloc[:, value_column].values.reshape(n_ts, 1, ts_sz) 
        y_train = np.array(outputs)

        # encode labels and convert to categorical
        self._label_encoder = LabelEncoder()
        y_ind = self._label_encoder.fit_transform(y_train.ravel())

        # calculate inverse class weights
        counts = np.bincount(y_ind).astype(np.float32)
        weights = [count / sum(counts) for count in counts]
        self._class_weights = [1 / w for w in weights]

        # convert labels to categorical
        n_classes = len(np.unique(y_ind))
        self._y_train = to_categorical(y_ind, n_classes)

        # instantiate classifier
        self._clf = generate_lstmfcn(
            ts_sz,
            n_classes,
            lstm_dim=self.hyperparams['lstm_dim'],
            attention=self.hyperparams['attention_lstm'],
            dropout=self.hyperparams['dropout_rate'])

        # model compilation and training
        self._clf.compile(optimizer = Adam(lr=self.hyperparams['learning_rate']), 
                         loss = 'categorical_crossentropy', 
                         metrics=['acc'])
        #self._clf.summary(print_fn = lambda x: print(x, file=sys.__stdout__))

        # save weights so we can start fitting from scratch (if desired by caller)
        self._clf.save_weights('model_initial_weights.h5')

        # mark that new training data has been set
        self._new_train_data = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits LSTM_FCN classifier using training data from set_training_data and hyperparameters
        '''

        # restore initial model weights if new training data
        if self._new_train_data:
            self._clf.load_weights('model_initial_weights.h5')

        # break out validation set if iterations arg not set
        if iterations is None:
            iterations_set = False
            train_split = 1 - self.hyperparams['val_split'] * self._X_train.shape[0]
            x_train = self._X_train[:int(train_split)].astype('float32')
            y_train = self._y_train[:int(train_split)].astype('float32')
            x_val = self._X_train[int(train_split):]
            y_val = self._y_train[int(train_split):]
            val_dataset = LSTMSequence(x_val, y_val, self.hyperparams['batch_size'])
            iterations = self.hyperparams['epochs']
            callbacks = [EarlyStopping(monitor='val_loss', 
                patience=self.hyperparams['early_stopping_patience'], 
                mode = 'min',
                restore_best_weights=False)]   
        else:
            iterations_set = True
            x_train = self._X_train
            y_train = self._y_train
            val_dataset = None
            monitor_quantity = 'loss'
            callbacks = None
            has_finished = False
        train_dataset = LSTMSequence(x_train, y_train, self.hyperparams['batch_size'])

        # time training for 1 epoch so we can consider timeout argument thoughtfully
        if timeout:
            logger.info('Timing the fitting procedure for one epoch so we can consider timeout thoughtfully')
            start_time = time.time()
            self._clf.fit(train_dataset,
                epochs = iterations, 
                validation_data = val_dataset,
                class_weight = self._class_weights,
                shuffle = True,
                use_multiprocessing = self.hyperparams['use_multiprocessing'],
                workers = self.hyperparams['num_workers'])
            epoch_time_estimate = time.time() - start_time
            timeout_epochs = timeout // epoch_time_estimate - 1 # subract 1 more to be safe
            iters = min(timeout_epochs, iterations)
            start_epoch = 1 # account for one training epoch that already happened for timing purposes
        else:
            iters = iterations
            start_epoch = 0

        # normal fitting
        logger.info(f'Fitting for {iterations-start_epoch} iterations')
        start_time = time.time()
        fitting_history = self._clf.fit(train_dataset, 
            epochs = iters, 
            validation_data = val_dataset,
            class_weight = self._class_weights,
            shuffle = True,
            use_multiprocessing = self.hyperparams['use_multiprocessing'],
            workers = self.hyperparams['num_workers'],
            callbacks = callbacks, 
            initial_epoch = start_epoch)
        iterations_completed = len(fitting_history.history['loss'])
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
        value_column, target_column, grouping_column = self._get_cols(inputs.metadata)

        n_ts = inputs.iloc[:, grouping_column[0]].nunique()
        ts_sz = inputs.shape[0] // n_ts
        x_vals = inputs.iloc[:, value_column].values.reshape(n_ts, 1, ts_sz)
        test_dataset = LSTMSequenceTest(x_vals, self.hyperparams['batch_size'])

        # make predictions
        preds = self._clf.predict_generator(test_dataset, 
            use_multiprocessing = self.hyperparams['use_multiprocessing'],
            workers = self.hyperparams['num_workers'])
        preds = self._label_encoder.inverse_transform(np.argmax(preds, axis = 1))

        # create output frame
        result_df = container.DataFrame({inputs.columns[target_column[0]]: preds}, generate_metadata=True)
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 
            ('https://metadata.datadrivendiscovery.org/types/PredictedTarget'))

        # ok to set to True because we have checked that primitive has been fit
        return CallResult(result_df, has_finished=True)


