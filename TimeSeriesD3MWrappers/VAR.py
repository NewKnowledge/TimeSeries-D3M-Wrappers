import sys
import os.path
import numpy as np
import pandas

from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):  
    pass

class Storc(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Produce primitive's best guess for the cluster number of each series.
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "76b5a479-c209-4d94-92b5-7eba7a4d4499",
        'version': __version__,
        'name': "VAR",
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
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),)
            }
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_forecasting.vector_autoregression.VAR',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.VECTOR_AUTOREGRESSION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._params = {}
        self._X_train = None          # training inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits Kmeans clustering algorithm using training data from set_training_data and hyperparameters
        '''
        self._kmeans.fit(self._X_train)
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
        inputs: numpy ndarray of size (number_of_time_series, time_series_length) containing training time series
        
        '''
        # use all attribute columns (and suggested target!) as training data

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : Input pandas frame where each row is a series.  Series timestamps are store in the column names.

        Returns
        -------
        Outputs
            The output is a dataframe containing a single column where each entry is the associated series' cluster number.
        """
        # split filenames into d3mIndex (hacky)
        filename_index = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        col_name = inputs.metadata.query_column(filename_index[0])['name']
        d3mIndex_df = pandas.DataFrame([int(filename.split('_')[0]) for filename in inputs[col_name]])

        ts_loader = TimeSeriesLoaderPrimitive(hyperparams = {"time_col_index":0, "value_col_index":1, "file_col_index": None})
        inputs = ts_loader.produce(inputs = inputs).value

        # set number of clusters for k-means
        labels = self._kmeans.predict(inputs.values)
        
        # add metadata to output
        labels = pandas.DataFrame(labels)
        out_df_sloth = pandas.concat([d3mIndex_df, labels], axis = 1)
        sloth_df = d3m_DataFrame(out_df_sloth)
        
        # first column ('d3mIndex')
        col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        
        # second column ('labels')
        col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        # concatentate final output frame -- not real consensus from program, so commenting out for now
        # out_df = utils_cp.append_columns(out_df, sloth_df)

        return CallResult(sloth_df)

if __name__ == '__main__':
    
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"0"}))
    df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)
    hyperparams_class = Storc.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    storc_client = Storc(hyperparams = hyperparams_class.defaults().replace({'algorithm':'TimeSeriesKMeans','nclusters':4}))
    result = storc_client.produce(inputs = df)
    print(result.value)
    result.value.to_csv('sloth_predictions.csv', index = False)
    
