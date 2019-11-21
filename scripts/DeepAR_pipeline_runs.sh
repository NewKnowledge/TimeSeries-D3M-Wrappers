#!/bin/bash -e 

Datasets=('56_sunspots' '56_sunspots_monthly' 'LL1_736_population_spawn' 'LL1_736_population_spawn_simpler' 'LL1_736_stock_market' 'LL1_terra_canopy_height_long_form_s4_100' 'LL1_terra_canopy_height_long_form_s4_90' 'LL1_terra_canopy_height_long_form_s4_80' 'LL1_terra_canopy_height_long_form_s4_70' 'LL1_terra_leaf_angle_mean_long_form_s4')
cd /primitives
# git pull upstream master
# git checkout forecasting_pipelines
cd /primitives/v2019.11.10/Distil/d3m.primitives.time_series_forecasting.convolutional_neural_net.DeepAR/1.0.0
mkdir pipelines
cd pipelines
python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/forecasting_pipeline_imputer.py"
cd ..
mkdir pipeline_runs
cd pipeline_runs

#create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do
  
  # file="/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Sloth_pipeline_$i.py"
  # match="step_1.add_output('produce')"
  # insert="step_1.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')"
  # sed -i "/step_1.add_argument(name='outputs'/d" $file
  # sed -i "s/$match/$match\n$insert/" $file
  # # generate and save pipeline + metafile

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}_validation.yml
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # # cleanup temporary file
  rm scores.csv
done

# zip pipeline runs individually
# cd ..
# gzip -r pipeline_runs