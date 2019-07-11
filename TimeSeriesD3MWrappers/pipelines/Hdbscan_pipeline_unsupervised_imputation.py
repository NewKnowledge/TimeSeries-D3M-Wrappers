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

# Step 1, 2: Extract target columns from dataset
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
step_1.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,data=('https://metadata.datadrivendiscovery.org/types/Target',))
pipeline_description.add_step(step_1)

step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_2.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=1)
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3, 4: Add Integer type to target column
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.add_semantic_types.DataFrameCommon'))
step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,data=('http://schema.org/Integer',))
pipeline_description.add_step(step_3)

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_4.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=3)
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5, 6: Extract attribute columns
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
pipeline_description.add_step(step_5)

step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_6.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=5)
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Step 7, 8: Append attribute columns and target columns
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.horizontal_concat.DataFrameCommon'))
pipeline_description.add_step(step_7)

step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_8.add_argument(name='left', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_8.add_argument(name='right', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_8.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=7)
step_8.add_output('produce')
pipeline_description.add_step(step_8)

# Step 9, 10: column parser
step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.DataFrameCommon'))
pipeline_description.add_step(step_9)

step_10 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_10.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.8.produce')
step_10.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=9)
step_10.add_output('produce')
pipeline_description.add_step(step_10)

# Step 11,12: imputer
step_11 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
step_11.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
step_11.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True)
pipeline_description.add_step(step_11)

step_12 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
step_12.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.10.produce')
step_12.add_hyperparameter(name='primitive', argument_type= ArgumentType.PRIMITIVE, data=11)
step_12.add_output('produce')
pipeline_description.add_step(step_12)

# Step 13: DISTIL/NK Storc primitive
step_13 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.clustering.hdbscan.Hdbscan'))
step_13.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.12.produce')
step_13.add_hyperparameter(name='long_format', argument_type= ArgumentType.VALUE, data=True)
step_13.add_output('produce')
pipeline_description.add_step(step_13)

# Step 14,15,16: Distil ensemble classifier
step_14 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
step_14.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.13.produce')
step_14.add_output('produce')
pipeline_description.add_step(step_14)

step_15 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
step_15.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.13.produce')
step_15.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,data=('https://metadata.datadrivendiscovery.org/types/Target',))
step_15.add_output('produce')
pipeline_description.add_step(step_15)

step_16 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.xgboost_gbtree.DataFrameCommon'))
step_16.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.14.produce')
step_16.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.15.produce')
step_16.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
step_16.add_output('produce')
pipeline_description.add_step(step_16)

# Step 13: construct output
step_17 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon'))
step_17.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.16.produce')
step_17.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.13.produce')
step_17.add_output('produce')
pipeline_description.add_step(step_17)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.17.produce')

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
