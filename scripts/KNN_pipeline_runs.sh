#!/bin/bash -e 

Datasets=('66_chlorineConcentration_MIN_METADATA' 'LL1_Adiac_MIN_METADATA' 'LL1_ArrowHead_MIN_METADATA' 'LL1_CinC_ECG_torso_MIN_METADATA' 'LL1_Cricket_Y_MIN_METADATA' 'LL1_ECG200_MIN_METADATA' 'LL1_ElectricDevices_MIN_METADATA' 'LL1_FISH_MIN_METADATA' 'LL1_FaceFour_MIN_METADATA' 'LL1_FordA_MIN_METADATA' 'LL1_HandOutlines_MIN_METADATA' 'LL1_Haptics_MIN_METADATA' 'LL1_ItalyPowerDemand_MIN_METADATA' 'LL1_Meat_MIN_METADATA' 'LL1_OSULeaf_MIN_METADATA')
# cd /primitives/v2019.11.10/Distil/d3m.primitives.time_series_classification.k_neighbors.Kanine/1.0.3/pipelines
# python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Kanine_pipeline.py"
# cd ..
# mkdir pipeline_runs
# cd pipeline_runs

for i in "${Datasets[@]}"; do

  # generate pipeline run
  python3 -m d3m runtime -d /datasets/ fit-score -p pipeline.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json #-O $i.yml

done

# # zip pipeline runs individually
# cd ..
# gzip -r pipeline_runs
