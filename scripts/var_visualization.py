# compare ARIMA and VAR predictions on examples from population spawn dataset
from common_primitives import dataset_to_dataframe as DatasetToDataFrame
from d3m import container
from d3m.primitives.time_series_forecasting.vector_autoregression import VAR
from Sloth.predict import Arima
from d3m.container import DataFrame as d3m_DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error as mae
# dataset to dataframe
input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/LL1_736_population_spawn_simpler/TRAIN/dataset_TRAIN/datasetDoc.json')
hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)
original = df.copy()

# apply VAR to predict each species in each sector
n_periods = 25
var_hp = VAR.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
var = VAR(hyperparams = var_hp.defaults().replace({'filter_index_two':1, 'filter_index_one':2, 'n_periods':n_periods, 'interval':25, 'datetime_index_unit':'D'}))
var.set_training_data(inputs = df, outputs = None)
var.fit()
test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/LL1_736_population_spawn_simpler/TEST/dataset_TEST/datasetDoc.json')
test_df = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value)
test_df = test_df.drop(columns = 'count')
var_pred = var.produce(inputs = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value)).value
var_pred = var_pred.merge(test_df, on = 'd3mIndex', how='left')

# load targets data
targets = pd.read_csv('file:///datasets/seed_datasets_current/LL1_736_population_spawn_simpler/SCORE/targets.csv')
test_df['d3mIndex'] = test_df['d3mIndex'].astype(int)
targets = targets.merge(test_df,on = 'd3mIndex', how = 'left')

# # compare VAR predictions to ARIMA predictions for individual species / sectors
# sector = 'S_3102'
# species = ['cas9_VBBA', 'cas9_FAB', 'cas9_JAC', 'cas9_CAD', 'cas9_YABE']
# original = original[original['sector'] == sector]
# var_pred = var_pred[var_pred['sector'] == sector]
# targets = targets[targets['sector'] == sector]

# # instantiate arima primitive
# clf = Arima(True)

# for specie in species:
    
#     x_train = original[original['species'] == specie]['day'].values.astype(int)
#     train = original[original['species'] == specie]['count'].values.astype(float)
#     v_pred = var_pred[var_pred['species'] == specie]['count'].values.astype(float)
#     true = targets[targets['species'] == specie]['count'].values.astype(float)
#     x_pred = var_pred[var_pred['species'] == specie]['day'].values.astype(int)

#     clf.fit(train)
#     a_pred = clf.predict(n_periods)[-1:]

#     # plot results
#     plt.clf()
#     plt.scatter(x_train, train, c = 'blue', label = 'true values')
#     plt.scatter(x_pred, true, c = 'blue', label = 'true values')
#     plt.scatter(x_pred, v_pred, c = 'green', label = 'VAR prediction', alpha = 0.5)
#     plt.scatter(x_pred, a_pred, c = 'red', label = 'ARIMA prediction', alpha = 0.5)
#     plt.xlabel('Days of the Year')
#     plt.ylabel('Species Count')
#     plt.title(f'VAR vs. ARIMA Comparison on Species {specie} in Sector {sector}')
#     plt.legend()
#     plt.savefig(f'{specie}.png')

# compare VAR predictions to ARIMA for each sector
clf = Arima(True)
#sectors = ['S_3102', 'S_4102', 'S_5102']
#species = ['cas9_VBBA', 'cas9_FAB', 'cas9_JAC', 'cas9_CAD', 'cas9_YABE', 'cas9_HNAF', 'cas9_NIAG', 'cas9_MBI']

COLORS = ["#FA5655", "#F79690", "#B9BC2D", "#86B6B2", "#955B99", "#252B7A"]
for sector in targets['sector'].unique():
    original_1 = original[original['sector'] == sector]
    var_pred_1 = var_pred[var_pred['sector'] == sector]
    print(var_pred_1)
    targets_1 = targets[targets['sector'] == sector]

    # arima prediction on each species in sector
    a_pred = []
    for specie in targets_1['species'].unique():
        train = original_1[original_1['species'] == specie]['count'].values.astype(float)
        clf.fit(train)
        a_pred = a_pred.append(clf.predict(n_periods)[-1:])

    print(f'mae: {mae(targets_1, a_pred)}')
    plt.scatter(sector, mae(targets_1, a_pred), c = COLORS[0], label = 'MAE of ARIMA prediction')
    plt.scatter(sector, mae(targets_1, v_pred_1), c = COLORS[1], label = 'MAE of VAR prediction')
plt.xlabel(f'Sector')
plt.ylabel('MAE')
plt.title(f'VAR vs. ARIMA MAE Comparison for Each Sector')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig(f'mae_comp.png')





