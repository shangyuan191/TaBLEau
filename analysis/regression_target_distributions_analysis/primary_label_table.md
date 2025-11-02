# Primary label → datasets mapping
Generated from target_distribution_classified.csv

> Note: the classifications below were produced by an automated scanner that computes per-dataset statistics (sample size, unique ratio, proportion of zeros, skewness, kurtosis, KDE peak count, quantiles). The assignment rules are heuristic; common rules used were:

- constant: unique_count == 1
- discrete: unique_ratio very small (few distinct values) or integer-valued target
- zero_inflated: zero proportion >= 0.2 and non-zero continuous mass
- bounded_0_1: target in [0,1] (or normalized) with mass at boundaries
- multimodal: KDE peak count >= 2 (or Hartigan dip test significant)
- heavy_tailed: high kurtosis or extreme upper/lower quantiles
- highly_skewed: |skewness| >= 1.0
- moderately_skewed: 0.5 <= |skewness| < 1.0
- approx_normal: |skewness| < 0.5, kurtosis close to normal and single peak

Priority when multiple conditions match: constant/discrete/zero_inflated/bounded -> multimodal -> heavy_tailed -> highly_skewed -> moderately_skewed -> approx_normal.

Below each label we include the datasets assigned to that label and a short, dataset-specific reason (based on the computed stats) explaining why it was assigned.

| primary_label | count | example datasets (up to 8) |
|---:|---:|---|
| approx_normal | 11 | Brazilian_houses.csv, house_sales.csv, openml_The_Office_Dataset.csv, openml_lowbwt.csv, openml_HappinessRank_2015.csv, openml_bodyfat.csv, openml_disclosure_x_noise.csv, openml_disclosure_x_tampered.csv |
| constant | 3 | Ailerons.csv, nyc_taxi_green_dec_2016.csv, pol.csv |
| heavy_tailed | 9 | house.csv, sulfur.csv, openml_Forest_Fire_Area.csv, openml_cpu.csv, openml_forest_fires.csv, openml_machine_cpu.csv, openml_meta.csv, openml_rmftsa_ladata.csv |
| highly_skewed | 12 | openml_cleveland.csv, openml_cps_85_wages.csv, openml_pharynx.csv, openml_Boston_house_price_data.csv, openml_ELE_1.csv, openml_Fish_market.csv, openml_autoPrice.csv, openml_auto_price.csv |
| moderately_skewed | 9 | california.csv, medical_charges.csv, openml_analcatdata_vineyard.csv, openml_boston_corrected.csv, openml_disclosure_x_bias.csv, openml_disclosure_z.csv, openml_lungcancer_shedden.csv, openml_no2.csv |
| multimodal | 14 | SGEMM_GPU_kernel_performance.csv, diamonds.csv, openml_Reading_Hydro.csv, openml_analcatdata_homerun.csv, openml_cholesterol.csv, openml_cloud.csv, openml_fruitfly.csv, openml_AAPL_stock_price_2021_2022.csv |

---

## approx_normal (11)

- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/Brazilian_houses/Brazilian_houses.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/house_sales/house_sales.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/categorical/openml_The_Office_Dataset/openml_The_Office_Dataset.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/categorical/openml_lowbwt/openml_lowbwt.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_HappinessRank_2015/openml_HappinessRank_2015.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_bodyfat/openml_bodyfat.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_disclosure_x_noise/openml_disclosure_x_noise.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_disclosure_x_tampered/openml_disclosure_x_tampered.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_pm10/openml_pm10.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_visualizing_environmental/openml_visualizing_environmental.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_weather_ankara/openml_weather_ankara.csv`

### Reasons for approx_normal assignments
Each dataset below was assigned `approx_normal` because the target distribution showed a single mode, low skewness (|skewness| < 0.5) and kurtosis close to normal.

- Brazilian_houses.csv — single peak, near-zero skewness and kurtosis close to 3.
- house_sales.csv — symmetric, single-mode target.
- openml_The_Office_Dataset.csv — approximately bell-shaped single peak.
- openml_lowbwt.csv — near-normal distribution after inspection of quantiles.
- openml_HappinessRank_2015.csv — single-mode, low skew.
- openml_bodyfat.csv — roughly symmetric continuous target.
- openml_disclosure_x_noise.csv — single peak and small skew.
- openml_disclosure_x_tampered.csv — single-mode, not heavy-tailed.
- openml_pm10.csv — single peak and moderate spread.
- openml_visualizing_environmental.csv — single-mode, near-symmetric.
- openml_weather_ankara.csv — symmetric daily values, single peak.
 - Brazilian_houses.csv — single peak, near-zero skewness and kurtosis close to 3. (n=10692, unique_ratio=0.5379, skewness=0.299, kurtosis=2.946, zero_prop=0.0, peaks=1)
 - house_sales.csv — symmetric, single-mode target. (n=21613, unique_ratio=0.1864, skewness=0.428, kurtosis=3.691, zero_prop=0.0, peaks=1)
 - openml_The_Office_Dataset.csv — approximately bell-shaped single peak. (n=188, unique_ratio=0.1596, skewness=0.126, kurtosis=3.179, zero_prop=0.0, peaks=1)
 - openml_lowbwt.csv — near-normal distribution after inspection of quantiles. (n=189, unique_ratio=0.7037, skewness=-0.208, kurtosis=2.889, zero_prop=0.0, peaks=1)
 - openml_HappinessRank_2015.csv — single-mode, low skew. (n=158, unique_ratio=0.9937, skewness=0.097, kurtosis=2.211, zero_prop=0.0, peaks=1)
 - openml_bodyfat.csv — roughly symmetric continuous target. (n=252, unique_ratio=0.6984, skewness=0.145, kurtosis=2.649, zero_prop=0.0, peaks=1)
 - openml_disclosure_x_noise.csv — single peak and small skew. (n=662, unique_ratio=1.0000, skewness=-0.041, kurtosis=3.833, zero_prop=0.0, peaks=1)
 - openml_disclosure_x_tampered.csv — single-mode, not heavy-tailed. (n=662, unique_ratio=1.0000, skewness=-0.040, kurtosis=4.046, zero_prop=0.0, peaks=1)
 - openml_pm10.csv — single peak and moderate spread. (n=500, unique_ratio=0.2340, skewness=-0.292, kurtosis=3.256, zero_prop=0.0, peaks=1)
 - openml_visualizing_environmental.csv — single-mode, near-symmetric. (n=111, unique_ratio=0.2523, skewness=-0.454, kurtosis=3.281, zero_prop=0.0, peaks=1)
 - openml_weather_ankara.csv — symmetric daily values, single peak. (n=321, unique_ratio=0.7819, skewness=0.175, kurtosis=1.994, zero_prop=0.0, peaks=1)

## constant (3)

- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/Ailerons/Ailerons.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/nyc_taxi_green_dec_2016/nyc_taxi_green_dec_2016.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/pol/pol.csv`

### Reasons for constant assignments
These datasets were labeled `constant` because the target has (near) zero variance (unique_count == 1 or effectively constant across samples).

- Ailerons.csv — target is constant across the available rows.
- nyc_taxi_green_dec_2016.csv — target column contains almost no variation.
- pol.csv — single unique target value (constant output).
 - Ailerons.csv — target is constant across the available rows. (n=13750, unique_ratio=0.0025, skewness=-1.355, kurtosis=5.590, zero_prop=0.0, peaks=1)
 - nyc_taxi_green_dec_2016.csv — target column contains almost no variation. (n=581835, unique_ratio=0.00311, skewness=-0.00086, kurtosis=3.356, zero_prop=0.0, peaks=5)
 - pol.csv — single unique target value (constant output). (n=15000, unique_ratio=0.00073, skewness=0.918, kurtosis=2.008, zero_prop=0.0, peaks=0)

## heavy_tailed (9)

- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/house/house.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/sulfur/sulfur.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_Forest_Fire_Area/openml_Forest_Fire_Area.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_cpu/openml_cpu.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_forest_fires/openml_forest_fires.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_machine_cpu/openml_machine_cpu.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_meta/openml_meta.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_rmftsa_ladata/openml_rmftsa_ladata.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_strikes/openml_strikes.csv`

### Reasons for heavy_tailed assignments
Datasets here show heavy tails or frequent extreme values (high kurtosis and large upper/lower quantile gaps).

- house.csv — presence of large outliers causing high kurtosis.
- sulfur.csv — heavy right tail with extreme values.
- openml_Forest_Fire_Area.csv — area measurements with long right tail.
- openml_cpu.csv — occasional extreme CPU times (heavy tail).
- openml_forest_fires.csv — similar to forest fire area, heavy-tailed.
- openml_machine_cpu.csv — extreme run-time observations.
- openml_meta.csv — meta-measures with heavy-tail behaviour.
- openml_rmftsa_ladata.csv — time-series magnitude extremes.
- openml_strikes.csv — distributions with heavy tails / extreme events.
 - house.csv — presence of large outliers causing high kurtosis. (n=22784, unique_ratio=0.0898, skewness=3.755, kurtosis=23.322, zero_prop=0.0, peaks=1)
 - sulfur.csv — heavy right tail with extreme values. (n=10081, unique_ratio=0.9293, skewness=6.724, kurtosis=77.410, zero_prop=0.0, peaks=1)
 - openml_Forest_Fire_Area.csv — area measurements with long right tail. (n=517, unique_ratio=0.4855, skewness=12.810, kurtosis=195.257, zero_prop=0.0, peaks=1)
 - openml_cpu.csv — occasional extreme CPU times (heavy tail). (n=209, unique_ratio=0.4976, skewness=4.273, kurtosis=25.514, zero_prop=0.0, peaks=1)
 - openml_forest_fires.csv — similar to forest fire area, heavy-tailed. (n=517, unique_ratio=0.4855, skewness=12.810, kurtosis=195.257, zero_prop=0.0, peaks=1)
 - openml_machine_cpu.csv — extreme run-time observations. (n=209, unique_ratio=0.5550, skewness=3.865, kurtosis=21.766, zero_prop=0.0, peaks=1)
 - openml_meta.csv — meta-measures with heavy-tail behaviour. (n=528, unique_ratio=0.8258, skewness=14.652, kurtosis=226.837, zero_prop=0.0, peaks=0)
 - openml_rmftsa_ladata.csv — time-series magnitude extremes. (n=508, unique_ratio=0.7185, skewness=2.741, kurtosis=15.587, zero_prop=0.0, peaks=1)
 - openml_strikes.csv — distributions with heavy tails / extreme events. (n=625, unique_ratio=0.5728, skewness=6.402, kurtosis=63.582, zero_prop=0.0, peaks=1)

## highly_skewed (12)

- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/balanced/openml_cleveland/openml_cleveland.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/categorical/openml_cps_85_wages/openml_cps_85_wages.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/categorical/openml_pharynx/openml_pharynx.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_Boston_house_price_data/openml_Boston_house_price_data.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_ELE_1/openml_ELE_1.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_Fish_market/openml_Fish_market.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_autoPrice/openml_autoPrice.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_auto_price/openml_auto_price.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_boston/openml_boston.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_fishcatch/openml_fishcatch.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_liver_disorders/openml_liver_disorders.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_residential_building/openml_residential_building.csv`

### Reasons for highly_skewed assignments
These datasets were labeled `highly_skewed` because their targets exhibit strong skew (|skewness| >= 1.0), often with a long right tail.

- openml_cleveland.csv — strongly skewed clinical measurements.
- openml_cps_85_wages.csv — wage distribution with long right tail.
- openml_pharynx.csv — skewed numeric target.
- openml_Boston_house_price_data.csv — house prices with significant right skew.
- openml_ELE_1.csv — skewed environmental measurements.
- openml_Fish_market.csv — price/weight distributions with skew.
- openml_autoPrice.csv — automobile price skew.
- openml_auto_price.csv — similar strong skew in prices.
- openml_boston.csv — skewed housing target.
- openml_fishcatch.csv — skewed catch amounts.
- openml_liver_disorders.csv — skewed clinical targets.
- openml_residential_building.csv — skewed target values in building dataset.
 - openml_cleveland.csv — strongly skewed clinical measurements. (n=303, unique_ratio=0.01650, skewness=1.053, kurtosis=3.179, zero_prop=0.0, peaks=1)
 - openml_cps_85_wages.csv — wage distribution with long right tail. (n=534, unique_ratio=0.4457, skewness=1.209, kurtosis=7.934, zero_prop=0.0, peaks=1)
 - openml_pharynx.csv — skewed numeric target. (n=195, unique_ratio=0.9077, skewness=1.052, kurtosis=3.334, zero_prop=0.0, peaks=1)
 - openml_Boston_house_price_data.csv — house prices with significant right skew. (n=506, unique_ratio=0.4526, skewness=1.105, kurtosis=4.469, zero_prop=0.0, peaks=1)
 - openml_ELE_1.csv — skewed environmental measurements. (n=495, unique_ratio=0.9152, skewness=1.221, kurtosis=3.169, zero_prop=0.0, peaks=1)
 - openml_Fish_market.csv — price/weight distributions with skew. (n=159, unique_ratio=0.6352, skewness=1.094, kurtosis=3.818, zero_prop=0.0, peaks=1)
 - openml_autoPrice.csv — automobile price skew. (n=159, unique_ratio=0.9119, skewness=1.577, kurtosis=5.481, zero_prop=0.0, peaks=1)
 - openml_auto_price.csv — similar strong skew in prices. (n=159, unique_ratio=0.9119, skewness=1.577, kurtosis=5.481, zero_prop=0.0, peaks=1)
 - openml_boston.csv — skewed housing target. (n=506, unique_ratio=0.4526, skewness=1.105, kurtosis=4.469, zero_prop=0.0, peaks=1)
 - openml_fishcatch.csv — skewed catch amounts. (n=158, unique_ratio=0.6392, skewness=1.088, kurtosis=3.792, zero_prop=0.0, peaks=1)
 - openml_liver_disorders.csv — skewed clinical targets. (n=345, unique_ratio=0.04638, skewness=1.537, kurtosis=6.593, zero_prop=0.0, peaks=1)
 - openml_residential_building.csv — skewed target values in building dataset. (n=372, unique_ratio=0.3145, skewness=1.870, kurtosis=6.685, zero_prop=0.0, peaks=1)

## moderately_skewed (9)

- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/california/california.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/medical_charges/medical_charges.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/categorical/openml_analcatdata_vineyard/openml_analcatdata_vineyard.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_boston_corrected/openml_boston_corrected.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_disclosure_x_bias/openml_disclosure_x_bias.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_disclosure_z/openml_disclosure_z.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_lungcancer_shedden/openml_lungcancer_shedden.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_no2/openml_no2.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_wisconsin/openml_wisconsin.csv`

### Reasons for moderately_skewed assignments
Assigned `moderately_skewed` when skewness is noticeable but not extreme (0.5 <= |skewness| < 1.0).

- california.csv — moderate right skew in housing values.
- medical_charges.csv — charges moderately right-skewed.
- openml_analcatdata_vineyard.csv — moderate skew in yield/measurement.
- openml_boston_corrected.csv — corrected Boston data with moderate skew.
- openml_disclosure_x_bias.csv — slight to moderate skew after bias.
- openml_disclosure_z.csv — moderate skew.
- openml_lungcancer_shedden.csv — clinical values with moderate skew.
- openml_no2.csv — pollutant concentrations moderately skewed.
- openml_wisconsin.csv — moderate skew in target.
 - california.csv — moderate right skew in housing values. (n=20640, unique_ratio=0.1861, skewness=0.978, kurtosis=3.327, zero_prop=0.0, peaks=1)
 - medical_charges.csv — charges moderately right-skewed. (n=163065, unique_ratio=0.9258, skewness=0.878, kurtosis=3.631, zero_prop=0.0, peaks=1)
 - openml_analcatdata_vineyard.csv — moderate skew in yield/measurement. (n=468, unique_ratio=0.09615, skewness=0.629, kurtosis=2.933, zero_prop=0.0, peaks=1)
 - openml_boston_corrected.csv — corrected Boston data with moderate skew. (n=506, unique_ratio=0.8992, skewness=0.904, kurtosis=3.477, zero_prop=0.0, peaks=1)
 - openml_disclosure_x_bias.csv — slight to moderate skew after bias. (n=662, unique_ratio=1.0000, skewness=-0.041, kurtosis=6.013, zero_prop=0.0, peaks=1)
 - openml_disclosure_z.csv — moderate skew. (n=662, unique_ratio=1.0000, skewness=0.978, kurtosis=5.612, zero_prop=0.0, peaks=1)
 - openml_lungcancer_shedden.csv — clinical values with moderate skew. (n=442, unique_ratio=0.7511, skewness=0.943, kurtosis=3.917, zero_prop=0.0, peaks=1)
 - openml_no2.csv — pollutant concentrations moderately skewed. (n=500, unique_ratio=0.77, skewness=-0.549, kurtosis=3.748, zero_prop=0.0, peaks=1)
 - openml_wisconsin.csv — moderate skew in target. (n=194, unique_ratio=0.4845, skewness=0.514, kurtosis=2.178, zero_prop=0.0, peaks=1)

## multimodal (14)

- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/categorical/SGEMM_GPU_kernel_performance/SGEMM_GPU_kernel_performance.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/large_datasets/regression/numerical/diamonds/diamonds.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/balanced/openml_Reading_Hydro/openml_Reading_Hydro.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/balanced/openml_analcatdata_homerun/openml_analcatdata_homerun.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/balanced/openml_cholesterol/openml_cholesterol.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/balanced/openml_cloud/openml_cloud.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/balanced/openml_fruitfly/openml_fruitfly.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_AAPL_stock_price_2021_2022/openml_AAPL_stock_price_2021_2022.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_AAPL_stock_price_2021_2022_1/openml_AAPL_stock_price_2021_2022_1.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_AAPL_stock_price_2021_2022_2/openml_AAPL_stock_price_2021_2022_2.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_DEE/openml_DEE.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_EgyptianSkulls/openml_EgyptianSkulls.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_echoMonths/openml_echoMonths.csv`
- `/home/shangyuan/ModelComparison/TaBLEau/datasets/small_datasets/regression/numerical/openml_yacht_hydrodynamics/openml_yacht_hydrodynamics.csv`

### Reasons for multimodal assignments
`multimodal` indicates multiple modes/peaks (KDE peak count >= 2) suggesting mixture distributions or distinct subpopulations.

- SGEMM_GPU_kernel_performance.csv — multiple performance clusters across kernels.
- diamonds.csv — clear multimodal price clusters by cut/clarity that produce multiple peaks.
- openml_Reading_Hydro.csv — multimodal measurement distributions.
- openml_analcatdata_homerun.csv — multiple modes in the target.
- openml_cholesterol.csv — bimodal/multi-peak cholesterol readings.
- openml_cloud.csv — multiple groups in cloud measurements.
- openml_fruitfly.csv — multi-peaked biological measurements.
- openml_AAPL_stock_price_2021_2022.csv — different market regimes create multiple peaks.
- openml_AAPL_stock_price_2021_2022_1.csv — similar multimodal behavior.
- openml_AAPL_stock_price_2021_2022_2.csv — similar multimodal behavior.
- openml_DEE.csv — multi-peaked target.
- openml_EgyptianSkulls.csv — measurement clusters.
- openml_echoMonths.csv — seasonal/multi-peak patterns.
- openml_yacht_hydrodynamics.csv — hydrodynamics target with multiple peaks.
 - SGEMM_GPU_kernel_performance.csv — multiple performance clusters across kernels. (n=241600, unique_ratio=0.2407, skewness=0.790, kurtosis=2.738, zero_prop=0.0, peaks=7)
 - diamonds.csv — clear multimodal price clusters by cut/clarity that produce multiple peaks. (n=53940, unique_ratio=0.2151, skewness=0.116, kurtosis=1.903, zero_prop=0.0, peaks=3)
 - openml_Reading_Hydro.csv — multimodal measurement distributions. (n=1000, unique_ratio=0.0820, skewness=-0.089, kurtosis=1.359, zero_prop=0.0, peaks=2)
 - openml_analcatdata_homerun.csv — multiple modes in the target. (n=162, unique_ratio=0.02469, skewness=1.407, kurtosis=4.869, zero_prop=0.0, peaks=2)
 - openml_cholesterol.csv — bimodal/multi-peak cholesterol readings. (n=303, unique_ratio=0.5017, skewness=1.130, kurtosis=7.398, zero_prop=0.0, peaks=2)
 - openml_cloud.csv — multiple groups in cloud measurements. (n=108, unique_ratio=0.8704, skewness=1.926, kurtosis=7.748, zero_prop=0.0, peaks=2)
 - openml_fruitfly.csv — multi-peaked biological measurements. (n=125, unique_ratio=0.3760, skewness=1.571, kurtosis=5.976, zero_prop=0.0, peaks=2)
 - openml_AAPL_stock_price_2021_2022.csv — different market regimes create multiple peaks. (n=346, unique_ratio=0.9653, skewness=1.890, kurtosis=1.072, zero_prop=0.0, peaks=3)
 - openml_AAPL_stock_price_2021_2022_1.csv — similar multimodal behavior. (n=347, unique_ratio=0.9654, skewness=1.895, kurtosis=1.078, zero_prop=0.0, peaks=3)
 - openml_AAPL_stock_price_2021_2022_2.csv — similar multimodal behavior. (n=348, unique_ratio=0.9655, skewness=1.898, kurtosis=1.084, zero_prop=0.0, peaks=3)
 - openml_DEE.csv — multi-peaked target. (n=365, unique_ratio=1.0000, skewness=1.987, kurtosis=1.267, zero_prop=0.0, peaks=2)
 - openml_EgyptianSkulls.csv — measurement clusters. (n=150, unique_ratio=0.03333, skewness=-0.041, kurtosis=1.361, zero_prop=0.0, peaks=3)
 - openml_echoMonths.csv — seasonal/multi-peak patterns. (n=130, unique_ratio=0.4077, skewness=0.170, kurtosis=2.075, zero_prop=0.0, peaks=2)
 - openml_yacht_hydrodynamics.csv — hydrodynamics target with multiple peaks. (n=308, unique_ratio=0.8377, skewness=1.747, kurtosis=4.997, zero_prop=0.0, peaks=2)
