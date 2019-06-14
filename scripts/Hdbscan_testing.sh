#!/bin/bash -e 

#cd /
#git clone https://gitlab.com/jgleason/primitives
cd /primitives
#git branch lstm_pipelines
git checkout lstm_pipelines
#git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
#git pull upstream master

#Datasets=('LL1_Adiac' 'LL1_ArrowHead' '66_chlorineConcentration' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf')
Datasets=('SEMI_1040_sylva_prior' 'SEMI_1217_click_prediction_small')
rm /primitives/v2019.6.7/Distil/d3m.primitives.clustering.hdbscan.Hdbscan/1.0.2/pipelines/test_pipeline/*.meta
rm /primitives/v2019.6.7/Distil/d3m.primitives.clustering.hdbscan.Hdbscan/1.0.2/pipelines/test_pipeline/*.json
cd /primitives/v2019.6.7/Distil/d3m.primitives.clustering.hdbscan.Hdbscan/1.0.2/pipelines
#mkdir test_pipeline
cd test_pipeline

# create text file to record scores and timing information
#touch scores.txt
#echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Hdbscan_pipeline.py" $i

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
