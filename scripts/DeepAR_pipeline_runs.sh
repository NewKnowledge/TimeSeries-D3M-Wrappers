#!/bin/bash -e 

Datasets=('56_sunspots_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA')
#Datasets=('56_sunspots' '56_sunspots_monthly' 'LL1_736_population_spawn' 'LL1_736_population_spawn_simpler' 'LL1_736_stock_market' 'LL1_terra_canopy_height_long_form_s4_100' 'LL1_terra_canopy_height_long_form_s4_90' 'LL1_terra_canopy_height_long_form_s4_80' 'LL1_terra_canopy_height_long_form_s4_70' 'LL1_terra_leaf_angle_mean_long_form_s4' 'LL1_PHEM_Monthly_Malnutrition' 'LL1_PHEM_weeklyData_Malnutrition')
cd /primitives/v2020.1.9/Distil/d3m.primitives.time_series_forecasting.lstm.DeepAR/1.2.0/pipeline_runs

#create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/afc4caf6-8281-43c0-83b1-6062ba6ae0e5.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}.yml
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/0a4c2609-c8ca-4506-9c7a-b77a80eca62e.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}_force_count_data.yml
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/6c077b2b-1b56-47d9-88b3-cf44d80e1821.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}_winow_size_60.yml
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/10214465-b38c-4748-81cd-45b119fffd58.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}_one_window_context.yml
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

