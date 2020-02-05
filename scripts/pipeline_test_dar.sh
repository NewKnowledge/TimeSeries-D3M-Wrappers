#!/bin/bash -e 

Datasets=('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')
#Datasets=('56_sunspots_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA' 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA' 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA' 'LL1_PHEM_weeklyData_Malnutrition_MIN_METADATA')
for i in "${Datasets[@]}"; do

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -p pipeline_window.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json 
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

done

