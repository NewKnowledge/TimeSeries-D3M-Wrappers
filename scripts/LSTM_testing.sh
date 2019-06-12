#!/bin/bash  

Datasets=('LL1_Adiac' 'LL1_ArrowHead' '66_chlorineConcentration' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf'
cd /primitives/v2019.6.7/Distil/d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN/1.0.0/pipelines
mkdir test_pipeline
cd test_pipeline
for i in "${Datasets[@]}"; do
  python3 "LSTM_FCN_pipeline_$i.py" $i 
  python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json
  # logic to write score and timing information to a separate file
  cp * ../
  rm *
done
