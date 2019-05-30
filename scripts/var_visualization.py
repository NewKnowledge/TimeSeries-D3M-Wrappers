# compare ARIMA and VAR predictions on examples from population spawn dataset
from common_primitives import dataset_to_dataframe as DatasetToDataFrame
from d3m import container
from d3m.primitives.time_series_forecasting.vector_autoregression import VAR
from Sloth.predict import Arima
import pandas as pd
import matplotlib.pyplot as plt

# dataset to dataframe
input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/LL1_736_population_spawn_simpler/TRAIN/dataset_TRAIN/datasetDoc.json')
hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)

# apply VAR to predict each species in each sector
n_periods = 25
var_hp = VAR.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
var = VAR(hyperparams = var_hp.defaults().replace({'filter_index_two':1, 'filter_index_one':2, 'n_periods':n_periods, 'interval':25, 'datetime_index_unit':'D'}))
var.set_training_data(inputs = df, outputs = None)
var.fit()
test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/LL1_736_population_spawn_simpler/TEST/dataset_TEST/datasetDoc.json')
test_df = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value)
test_df.set_index('d3mIndex')
var_pred = var.produce(inputs = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value)).value
var_pred.set_index('d3mIndex')
var_pred = var_pred.join(test_df)

# load targets data
targets = pd.read_csv('file:///datasets/seed_datasets_current/LL1_736_population_spawn_simpler/SCORE/targets.csv')
targets.set_index('d3mIndex')
targets = targets.join(test_df)

# compare VAR predictions to ARIMA predictions
sector = 'S_3102'
species = ['cas9_VBBA']#, 'cas9_FAB', 'cas9_JAC', 'cas9_CAD', 'cas9_YABE']
original = df[df['sector'] == sector]
var_pred = var_pred[var_pred['sector'] == sector]
targets = var_pred[var_pred]['sector'] == sector]

# instantiate arima primitive
clf = Arima(True)

for specie in species:
    
    x_train = original[original['species'] == specie]['day'].values
    train = original[original['species'] == specie]['count'].values
    v_pred = var_pred[var_pred['species'] == specie]['count'].values
    true = targets[targets['species'] == specie]['count'].values
    x_pred = var_pred[var_pred['species'] == specie]['day'].values

    clf.fit(train)
    a_pred = clf.predict(n_periods)[-1:]

    # plot results
    plt.clf()
    plt.scatter(x_train, train, c = 'blue', label = 'true values')
    plt.scatter(x_pred, true, c = 'blue', label = 'true values')
    plt.scatter(x_pred, v_pred, c = 'green', label = 'VAR prediction')
    plt.scatter(x_pred, a_pred, c = 'red', label = 'ARIMA prediction')
    plt.xlabel('Days of the Year')
    plt.ylabel('Species Count')
    plt.title(f'VAR vs. ARIMA Comparison on Species {specie} in Sector {sector}')
    plt.show()



