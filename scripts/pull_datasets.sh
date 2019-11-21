Datasets=('LL1_Adiac' 'LL1_ArrowHead' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf' '66_chlorineConcentration' '56_sunspots' '56_sunspots_monthly' 'LL1_736_population_spawn' 'LL1_736_population_spawn_simpler' 'LL1_736_stock_market' 'LL1_terra_canopy_height_long_form_s4_100' 'LL1_terra_canopy_height_long_form_s4_90' 'LL1_terra_canopy_height_long_form_s4_80' 'LL1_terra_canopy_height_long_form_s4_70' 'LL1_terra_leaf_angle_mean_long_form_s4')

for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done