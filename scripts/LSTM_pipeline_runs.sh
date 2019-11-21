#!/bin/bash -e 

Datasets=('LL1_Adiac' , 'LL1_ArrowHead' , 'LL1_CinC_ECG_torso' , 'LL1_Cricket_Y' , 'LL1_ECG200' , 'LL1_ElectricDevices' , 'LL1_FISH' , 'LL1_FaceFour' , 'LL1_FordA' , 'LL1_HandOutlines' , 'LL1_Haptics' , 'LL1_ItalyPowerDemand' , 'LL1_Meat' , 'LL1_OSULeaf' , '66_chlorineConcentration')
cd /primitives
git pull upstream master
git checkout classification_pipelines
cd /primitives/v2019.11.10/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN/1.0.2
mkdir pipelines
cd pipelines
python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/LSTM_FCN_pipeline.py"
cd ..
mkdir pipeline_runs
cd pipeline_runs

#create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

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
