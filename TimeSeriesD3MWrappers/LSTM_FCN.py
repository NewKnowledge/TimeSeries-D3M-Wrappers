import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, concatenate, Activation, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from .layer_utils import AttentionLSTM
from sklearn.preprocessing import LabelEncoder

from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'


Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    attention_lstm = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
       description="whether to use attention in the lstm component of the model")
    lstm_cells = hyperparams.UniformInt(lower = 8, upper = 128, default = 128, 
        upper_inclusive = True, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of cells to use in the lstm component of the model')
    epochs = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 2000, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of training epochs')
    learning_rate = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 1e-3, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    batch_size = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 128, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of training epochs')
    long_format = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether the input dataset is already formatted in long format or not")
    pass


class LSTM_FCN(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies a LSTM FCN (LSTM fully convolutional network) for time
        series classification. The implementation is based off this paper: 
        https://ieeexplore.ieee.org/document/8141873 and this base library: 
        https://github.com/NewKnowledge/LSTM-FCN.
    
        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "a55cef3a-a7a9-411e-9dde-5c935ff3504b",
        'version': __version__,
        'name': "lstm_fcn",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series', 'convolutional neural network', 'lstm', 'time series classification'],
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
        
        self.n_classes = None
        self._X_train = None          # training inputs
        self._y_train = None          # training labels  
        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self._hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})
        self.clf = None
        self.label_encoder = None

    def _generate_lstmfcn(self, MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS):

        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

        x = LSTM(NUM_CELLS)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(NB_CLASS, activation='softmax')(x)

        model = Model(ip, out)

        model.summary()

        return model

    def _generate_alstmfcn(self, MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS):

        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

        x = AttentionLSTM(NUM_CELLS)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)
        x = concatenate([x, y])
        out = Dense(NB_CLASS, activation='softmax')(x)
        model = Model(ip, out)
        return model

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits Shapelet classifier using training data from set_training_data and hyperparameters
        '''
        self.label_encoder = LabelEncoder()
        y_ind = self.label_encoder.fit_transform(self._y_train.ravel())
        recip_freq = len(self._y_train) / (len(self.label_encoder.classes_) * np.bincount(y_ind).astype(np.float64))
        class_weight = recip_freq[self.label_encoder.transform(np.unique(self._y_train))]
        y_ind = to_categorical(y_ind, len(np.unique(y_ind)))
        
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=1. / np.cbrt(2), cooldown=0, min_lr=1e-4, verbose=2)

        # model compilation and training
        self.clf.compile(optimizer = Adam(lr=self.hyperparams['learning_rate']), 
                         loss = 'categorical_crossentropy', 
                         metrics=['accuracy'])
        self.clf.fit(self._X_train, y_ind, 
                     batch_size = self.hyperparams['batch_size'], 
                     verbose = 0,
                     epochs = self.hyperparams['epochs'], 
                     class_weight = class_weight,
                     callbacks = [reduce_lr])
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
        self._X_train = np.array(inputs.value).reshape(n_ts, 1, ts_sz) 
        self._y_train = np.array(inputs.label.iloc[::ts_sz])
        if self.hyperparams['attention_lstm']:
            self.clf = self._generate_alstmfcn(ts_sz, len(np.unique(self._y_train)), self.hyperparams['lstm_cells'])
        else:
            self.clf = self._generate_lstmfcn(ts_sz, len(np.unique(self._y_train)), self.hyperparams['lstm_cells'])
    
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
        input_vals = np.array(inputs.value).reshape(n_ts, 1, ts_sz)

        # produce classifications using Shapelets
        classes = pandas.DataFrame(self.label_encoder.inverse_transform(np.argmax(self.clf.predict(input_vals), axis = 1)))
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
    hyperparams_class = LSTM_FCN.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'] 
    shallot_client = LSTM_FCN(hyperparams=hyperparams_class.defaults())
    shallot_client.set_training_data(inputs = input_dataset, outputs = None)
    shallot_client.fit()
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    results = shallot_client.produce(inputs = test_dataset)
    print(results.value)
