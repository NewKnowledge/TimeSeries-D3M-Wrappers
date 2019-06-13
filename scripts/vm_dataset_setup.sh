#!/bin/bash -e 

git lfs clone https://gitlab.datadrivendiscovery.org/d3m/datasets.git -X "*"
cd datasets

Datasets=('LL1_Adiac' 'LL1_ArrowHead' '66_chlorineConcentration' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf')

for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done

sudo docker pull registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7 
sudo docker run --rm -t -i --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/datasets,target=/datasets registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7 bash

pip3 install --upgrade -e git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers#egg=TimeSeriesD3MWrappers
git clone https://gitlab.com/jgleason/primitives
cd primitives
git checkout shallot_pipelines
git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
git pull upstream master