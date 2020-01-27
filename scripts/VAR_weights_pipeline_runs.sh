#!/bin/bash -e 

#Datasets=('56_sunspots_monthly_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA')
Datasets=('56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA' 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA' 'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')
#'56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA' 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA' 'LL1_PHEM_weeklyData_Malnutrition_MIN_METADATA')
# python3 /tswrap/TimeSeriesD3MWrappers/pipelines/var_pipeline_weights.py
# python3 /tswrap/TimeSeriesD3MWrappers/pipelines/var_pipeline_weights_interpret_pooling_max.py
# python3 /tswrap/TimeSeriesD3MWrappers/pipelines/var_pipeline_weights_interpret_value_series.py

for i in "${Datasets[@]}"; do

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-produce -p pipeline_ci_var_weights.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-produce -p pipeline_ci_var_weights_pooling.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i interpret pooling took $runtime----------"

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-produce -p pipeline_ci_var_weights_value.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i interpret value took $runtime----------"
done
rm pipeline_ci_var_weights.json
rm pipeline_ci_var_weights_pooling.json
rm pipeline_ci_var_weights_value.json

