#!/bin/bash -e 

Datasets=('LL1_Adiac' 'LL1_ArrowHead' '66_chlorineConcentration' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf')

cd ../TimeSeriesD3MWrappers/pipelines
for i in "${Datasets[@]}"; do
  cp "Sloth_pipeline.py" "Sloth_pipeline_$i.py"
done
rm "Sloth_pipeline.py"
