#!/bin/bash -e 

Datasets=('56_sunspots' '56_sunspots_monthly' 'LL1_736_population_spawn' 'LL1_736_population_spawn_simpler' 'LL1_736_stock_market' 'LL1_terra_canopy_height_long_form_s4_100' 'LL1_terra_canopy_height_long_form_s4_90' 'LL1_terra_canopy_height_long_form_s4_80' 'LL1_terra_canopy_height_long_form_s4_70' 'LL1_terra_leaf_angle_mean_long_form_s4' 'LL1_PHEM_Monthly_Malnutrition' 'LL1_PHEM_weeklyData_Malnutrition')
rm *.json
python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines//Users/jgleason/Documents/NewKnowledge/D3M/TimeSeries-D3M-Wrappers/TimeSeriesD3MWrappers/pipelines//Users/jgleason/Documents/NewKnowledge/D3M/TimeSeries-D3M-Wrappers/TimeSeriesD3MWrappers/pipelines/var_pipeline_confidence_intervals.py"

for i in "${Datasets[@]}"; do

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-produce -p *.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

done
