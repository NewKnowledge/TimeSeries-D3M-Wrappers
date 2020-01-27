#!/bin/bash -e 

Datasets=('66_chlorineConcentration_MIN_METADATA' 'LL1_Adiac_MIN_METADATA' 'LL1_ArrowHead_MIN_METADATA' 'LL1_CinC_ECG_torso_MIN_METADATA' 'LL1_Cricket_Y_MIN_METADATA' 'LL1_ECG200_MIN_METADATA' 'LL1_ElectricDevices_MIN_METADATA' 'LL1_FISH_MIN_METADATA' 'LL1_FaceFour_MIN_METADATA' 'LL1_FordA_MIN_METADATA' 'LL1_HandOutlines_MIN_METADATA' 'LL1_Haptics_MIN_METADATA' 'LL1_ItalyPowerDemand_MIN_METADATA' 'LL1_Meat_MIN_METADATA' 'LL1_OSULeaf_MIN_METADATA')
cd /primitives
# git pull upstream master
# git checkout classification_pipelines
cd /primitives/v2019.11.10/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN/1.0.2
# mkdir pipelines
cd pipelines
python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/LSTM_FCN_pipeline.py"
cd ..
# mkdir pipeline_runs
cd pipeline_runs

#create text file to record scores and timing information
# touch scores.txt
# echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate pipeline run and time
  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}_no_attention.yml
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
