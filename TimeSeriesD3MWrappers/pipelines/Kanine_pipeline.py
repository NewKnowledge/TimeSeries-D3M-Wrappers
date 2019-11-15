from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
# import sys

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: Denormalize
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: Ts formatter
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.data_transformation.data_cleaning.DistilTimeSeriesFormatter'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: DS to DF on formatted ts DS
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3: DS to DF on input DS
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_3.add_output('produce')
pipeline_description.add_step(step_3)

# step 4: column parser on input DF
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.DataFrameCommon'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5: parse target semantic types
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_hyperparameter(name='semantic_types', argument_type= ArgumentType.VALUE, data=["https://metadata.datadrivendiscovery.org/types/Target",
    "https://metadata.datadrivendiscovery.org/types/TrueTarget"])
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6: KNN
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.time_series_classification.k_neighbors.Kanine'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Step 7: construct predictions
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)

# output dataset metafile (from command line argument)
# metafile = blob[8:44] + '.meta'
# dataset = sys.argv[1]
# with open(metafile, 'w') as outfile:
#     outfile.write('{')
#     outfile.write(f'"problem": "{dataset}_problem",')
#     outfile.write(f'"full_inputs": ["{dataset}_dataset"],')
#     outfile.write(f'"train_inputs": ["{dataset}_dataset_TRAIN"],')
#     outfile.write(f'"test_inputs": ["{dataset}_dataset_TEST"],')
#     outfile.write(f'"score_inputs": ["{dataset}_dataset_SCORE"]')
#     outfile.write('}')