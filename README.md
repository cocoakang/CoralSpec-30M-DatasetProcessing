# CoralSpec-30M-DatasetProcessing
Scripts for mask generation and processing of the CoralSpec-30M coral spectral dataset.


## Generate Masks with Trained Networks

1. Download the dataset from the CoralSpec-30M data repository.  
   The full dataset consists of 1,286 folders, packaged into multiple zip archives.  
   You may download only the first zip file for testing or as a small-scale example.

2. Download the pretrained network models (`network_models.zip`) and extract them.  
   Place the extracted folder in the **same parent directory** as the dataset.
   
   An example directory structure is shown below:

<img src="figures/file_structure.png" width="300">

3. In `run.sh`, set the `DATASET_PATH` variable to this parent directory.

4. Run:
   ```bash
   bash run.sh
    ```

This will generate coral masks and store the processed results in the processed_data directory of each entry.
Logs and intermediate results will be stored under $DATASET_PATH/tensorboard_logs.

## Citation
If you find this dataset or code useful in your research, please cite:

(Citation information will be updated once the data repo is published.)