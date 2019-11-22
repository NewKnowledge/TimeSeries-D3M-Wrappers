#!/bin/bash -e 

#Datasets=('66_chlorineConcentration')
Datasets=('LL1_Adiac' 'LL1_ArrowHead' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf')# '66_chlorineConcentration')
cd /primitives/v2019.11.10/Distil/d3m.primitives.time_series_classification.k_neighbors.Kanine/1.0.3/pipelines
python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Kanine_pipeline.py"
cd ..
mkdir pipeline_runs
cd pipeline_runs

for i in "${Datasets[@]}"; do

  # generate pipeline run
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O $i.yml

done

# zip pipeline runs individually
cd ..
gzip -r pipeline_runs