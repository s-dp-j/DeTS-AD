# DeTS-AD
**DeTS-AD: Temporal-Structural Decoupling for Multivariate Time Series Anomaly Detection**

Anomaly evidence in multivariate time series is inherently heterogeneous, because abnormality can be reflected by mismatches in normal temporal evolution, inter-variable structural dependencies, or both. However, most existing methods encode these heterogeneous anomaly cues into a shared representation space and derive a unified anomaly score, where temporal and structural deviations may become entangled: temporal evolution mismatches can be weakened by dominant structural patterns, while structural dependency violations may remain hidden when indivi trajectories are locally reconstructible. To address this issue, we propose DeTS-AD, a temporal-structural decoupling framework that reformulates multivariate time series anomaly detection as the collaborative discrimination of temporal evolution mismatch and structural dependency mismatch. The temporal pathway models normal evolution mechanisms through frequency-enhanced representations that integrate global spectral contexts and multi-resolution frequency-local dynamics, and quantifies temporal evolution mismatch through time-domain reconstruction residuals. The structural pathway constructs state-adaptive normal structural references by modulating learned dependency prototypes with sample-level global structural representations, and measures structural dependency mismatch through deviations from these references. Furthermore, DeTS-AD introduces a distribution calibration and reliability-adaptive fusion mechanism to map heterogeneous anomaly evidence into a unified scoring space and generate sample-wise fusion weights according to pathway-specific normal fluctuation scales. Experiments on five public benchmark datasets and mechanism-level analyses validate the effectiveness of DeTS-AD in achieving robust anomaly detection and capturing complementary temporal and structural anomaly evidence.

## 📁 Dataset

- **SMAP**, **MSL**, and **SMD** datasets were obtained from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).
- **SWaT** dataset was obtained from the [TranAD repository](https://github.com/imperial-qore/TranAD).
- **PSM** dataset was obtained from the [DualTF repository](https://github.com/kaist-dmlab/DualTF).

Before running the code, please ensure that the datasets are organized under the expected directory structure (e.g., ./data/). 

---

## 🚀 Usage

```bash

python main.py \
    --framework DeTSAD \
    --dataset <dataset_names> \
    --win_size 100 \
    --data_path ./data \
    --input_c <Number of channels> \
    --output_c <Number of channels> \
    --d_model 128 \
    --temperature 0.1 \
    --anomaly_ratio 0.2 \
    --anomaly_score_method learnable_norm_weight \

Arguments:
--dataset: specify the dataset name, e.g., SMD, SWaT, SMAP, MSL, PSM.
--input_c: Number of input channels, e.g., 25, 55.
--output_c: Number of output channels, e.g., 25, 55.


## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
