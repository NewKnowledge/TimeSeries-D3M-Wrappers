#!/bin/bash -e 

Datasets=('LL1_Adiac' 'LL1_ArrowHead' '66_chlorineConcentration' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf')
mkdir /primitives/v2019.6.7/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN
mkdir /primitives/v2019.6.7/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN/1.0.0
mkdir /primitives/v2019.6.7/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN/1.0.0/pipelines
cd /primitives/v2019.6.7/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN/1.0.0/pipelines
mkdir test_pipeline
cd test_pipeline

# create text file to record scores and timing information
touch scores.txt
echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/LSTM_FCN_pipeline_$i.py" $i

  # test and score pipeline
  start=`date +%s` 
  python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour
  if [ $runtime -lt 3600 ]; then
     echo "$i took less than 1 hour, copying pipeline"
     cp * ../
  fi

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
done
