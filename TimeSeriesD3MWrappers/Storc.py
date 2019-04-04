import sys
import os.path
import numpy as np
import pandas

from Sloth.cluster import KMeans
from tslearn.datasets import CachedDatasets

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

from timeseriesloader.timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '2.0.3'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'GlobalAlignmentKernelKMeans', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['GlobalAlignmentKernelKMeans', 'TimeSeriesKMeans'],
        description = 'type of clustering algorithm to use')
    nclusters = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=3, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'number of clusters \
        to user in kernel kmeans algorithm')
    long_format = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether the input dataset is already formatted in long format or not")
    pass

class Storc(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Produce primitive's best guess for the cluster number of each series.
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "77bf4b92-2faa-3e38-bb7e-804131243a7f",
        'version': __version__,
        'name': "Sloth",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series','Clustering'],
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
        'python_path': 'd3m.primitives.clustering.k_means.Sloth',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.K_MEANS_CLUSTERING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._params = {}
        self._X_train = None          # training inputs
        self._kmeans = KMeans(self.hyperparams['nclusters'], self.hyperparams['algorithm'])

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
        if not self.hyperparams['long_format']:
            # temporary (until Uncharted adds conversion primitive to repo)
            inputs = TimeSeriesFormatterPrimitive().produce(inputs = inputs, file_index = 1, main_resource_index = 'learningData')
        else:
            hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
            ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
            inputs = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)

        # load and reshape training data
        # 'series_id' and 'value' should be set by metadata
        inputs = inputs.value
        n_ts = len(inputs['0'].series_id.unique())
        ts_sz = int(inputs['0'].value.shape[0] / len(inputs['0'].series_id.unique()))
        self._X_train = np.array(inputs['0'].value).reshape(n_ts, ts_sz, 1)

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
        # temporary (until Uncharted adds conversion primitive to repo)
        inputs = TimeSeriesFormatterPrimitive().produce(inputs = inputs, file_index = 1, main_resource_index = 'learningData')

        # parse values from output of time series formatter
        inputs = inputs.value
        n_ts = len(inputs['0'].series_id.unique())
        ts_sz = int(inputs['0'].value.shape[0] / len(inputs['0'].series_id.unique()))
        input_vals = np.array(inputs['0'].value).reshape(n_ts, ts_sz, 1)
        
        # concatenate predictions and d3mIndex
        labels = pandas.DataFrame(self._kmeans.predict(input_vals))
        # maybe change d3mIndex key here to be programatically generated 
        out_df_sloth = pandas.concat([pandas.DataFrame(inputs['0'].d3mIndex.unique()), labels], axis = 1)
        # get column names from metadata
        out_df_sloth.columns = ['d3mIndex', 'label']
        sloth_df = d3m_DataFrame(out_df_sloth)
        
        # first column ('d3mIndex')
        col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'd3mIndex'        
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        
        # second column ('labels')
        col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        return CallResult(sloth_df)

if __name__ == '__main__':
    
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = Storc.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    storc_client = Storc(hyperparams = hyperparams_class.defaults().replace({'algorithm':'TimeSeriesKMeans','nclusters':4}))
    storc_client.set_training_data(inputs = input_dataset, outputs = None)
    storc_client.fit()
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    results = storc_client.produce(inputs = test_dataset)
    print(results.value)
    
