#!/bin/bash -e 

#Datasets=('56_sunspots_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA' 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA' 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA' 'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')
Datasets=('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')
cd /primitives/v2020.1.9/Distil/d3m.primitives.time_series_forecasting.vector_autoregression.VAR/1.2.0/pipeline_runs

#create text file to record scores and timing information
# touch scores.txt
# echo "Lag Order 2" >> scores.txt
# echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do
  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}.yml
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

  # save information
  # IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  # echo "$i, $score, $runtime" >> scores.txt
  
  # # # cleanup temporary file
  # rm scores.csv
done

# # zip pipeline runs individually
# cd ..
# gzip -r pipeline_runs