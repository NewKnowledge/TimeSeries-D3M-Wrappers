from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: Denormalize primitive
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: dataset_to_dataframe
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_hyperparameter(name='dataframe_resource', argument_type= ArgumentType.VALUE, data='learningData')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: DISTIL/NK VAR primitive
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.time_series_forecasting.arima.Parrot'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_hyperparameter(name='filter_index_one', argument_type= ArgumentType.VALUE, data=2)
step_2.add_hyperparameter(name='filter_index_two', argument_type= ArgumentType.VALUE, data=1)
step_2.add_hyperparameter(name='n_periods', argument_type= ArgumentType.VALUE, data=58)
step_2.add_hyperparameter(name='specific_intervals', argument_type= ArgumentType.VALUE, 
           data = [[24, 54], [22, 54], [23, 56], [27, 57], [0, 0], [26, 56], [26, 55], [26, 52], [23, 55], [23, 56], [24, 56], [26, 56], [27, 53], [24, 56], [27, 56], [26, 52], [24, 54], [22, 54], [25, 57], [26, 52], [13, 54], [23, 57], [24, 56], [26, 56], [26, 55], [25, 55], [22, 54], [22, 56], [24, 56], [27, 51], [26, 52], [24, 54], [22, 54], [25, 57], [26, 56], [26, 55], [26, 52], [16, 55], [23, 56], [24, 56], [26, 56], [27, 53], [24, 54], [22, 54], [23, 56], [27, 57], [26, 55], [26, 52], [24, 54], [24, 57], [24, 56], [26, 56], [26, 55], [25, 55], [22, 54], [23, 56]])
step_2.add_hyperparameter(name='datetime_index_unit', argument_type= ArgumentType.VALUE, data='D')
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.2.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)

# output dataset metafile (from command line argument)
metafile = blob[8:44] + '.meta'
dataset = sys.argv[1]
with open(metafile, 'w') as outfile:
    outfile.write('{')
    outfile.write(f'"problem": "{dataset}_problem",')
    outfile.write(f'"full_inputs": ["{dataset}_dataset"],')
    outfile.write(f'"train_inputs": ["{dataset}_dataset_TRAIN"],')
    outfile.write(f'"test_inputs": ["{dataset}_dataset_TEST"],')
    outfile.write(f'"score_inputs": ["{dataset}_dataset_SCORE"]')
    outfile.write('}')
