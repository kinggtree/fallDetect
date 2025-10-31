# ActSyncDT: A Cloud-Edge Collaborative Healthcare Digital Twin Framework with Low-Cost and High-Fidelity Synchronization

## 模型运行顺序

1.  **`data_extract.py`**
    * **功能**: 提取数据到npy文件。
    * **产出文件**: `四个SensorXXXXSequences.npy文件`

2.  **`autoregression_feature_train.ipynb`**
    * **功能**: 训练并保存自回归模型。
    * **产出文件**: `autoregression_feature_extrator_model.pt`

3.  **`autoregression_feature_extract.ipynb`**
    * **功能**: 使用上一步的模型提取数据集的特征。
    * **产出文件**: `all_features.npy`, `all_labels.npy`

4.  **`fidelity_model_train_pretrained_encoder.ipynb`**
    * **功能**: 使用自回归模型的.pt文件（其中的encoder），训练特征和数据集以保存保真模型。
    * **产出文件**: `contextual_fidelity_model_pretrained_encoder.pth`

5.  **`center_cloud/model_runner.py`**
    * **功能**: 训练DQN采样模型。
    * **产出文件**: `dqn_agent_final.pth`

## 评估

> 最终结果与分析将在 Terminal 中显示。

* **`DQN_evaluate_cloud.py`**
    * **评估场景**: 当原始数据全部同步时。

* **`DQN_evaluate_lazySync.py`**
    * **评估场景**: 完全不同步任何原始数据。

* **`DQN_evaluate_random.py`**
    * **评估场景**: 以一个比例（超参数 `RANDOM_RATIO`）来随机同步原始数据。

* **`DQN_evaluate_uniform.py`**
    * **评估场景**: 以一个比例（超参数 `RATIO_OF_REAL_DATA`）来有规律地同步原始数据。

## 说明

* 根目录下的文件，使用 **MobiFall** 数据集来训练测试。
* 在 **UMAFall** 数据集上的训练和测试，请参见 `UMAFall_based` 文件夹（训练顺序同上）。