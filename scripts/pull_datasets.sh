cd ../../datasets
Datasets=('66_chlorineConcentration_MIN_METADATA' '56_sunspots_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA')
#'LL1_Adiac' 'LL1_ArrowHead' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf' 'LL1_terra_canopy_height_long_form_s4_90' 'LL1_terra_leaf_angle_mean_long_form_s4' 'LL1_terra_canopy_height_long_form_s4_70' 'LL1_PHEM_Monthly_Malnutrition' 'LL1_PHEM_weeklyData_Malnutrition'
for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done