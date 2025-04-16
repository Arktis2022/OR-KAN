OR-KAN: **Unsupervised Fetal Brain MRI Quality Control based on Slice-level Orientation Prediction Uncertainty using an Orientation Recognition KAN Model**



OR-KAN is a tool for quality control (QC) of T2-weighted (T2w) fetal brain MR images. 



#### Features

+ Training does not rely on labeled data.
+ Applicable to cross-device MRI scan data.
+ Equipped with orientation classification capabilities, it can be integrated into the slice-to-volume reconstruction pipeline (such as [NiftyMIC](https://www.sciencedirect.com/science/article/pii/S1053811919309152)).



#### Environment

+ Downloading `myenv.tar.gz` from Google Drive

+ Extract the package using the following command:  

  ```bash
  tar -xzf myenv.tar.gz -C YOUR_PATH/OR-KAN/conda_env
  ```

+ Activate the environment by running:  

  ```bash
  source YOUR_PATH/OR-KAN/conda_env/bin/activate
  ```



#### Usage

>  /checkpoint: Pre-trained weights for OR-KAN
>
> /data_example_with_mask: It includes one high-quality (from [here](https://zenodo.org/records/8123677)) and one low-quality (from [here](https://pubmed.ncbi.nlm.nih.gov/35082346/)) fetal brain MR image as examples for testing.

+ For brain extraction on your fetal brain images, it is recommended to use the [Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction) tool.

+ Run the quality control model with the following command:  

  ```bash
  python quality_control.py --input_dir YOUR_PATH/OR-KAN/data_example_with_mask
  ```

+ Run the quality control pipeline for reconstruction using the following command:  

  ```bash
  python quality_control_for_recon.py --input_dir YOUR_DATA --output_dir YOUR_OUTPUT_DIR
  ```

  In this command, `YOUR_DATA` should contain all T2-weighted scans (`.nii.gz` files) for a single subject. Upon completion, `YOUR_OUTPUT_DIR` will contain the highest-quality image for each of the three orientations.

#### Acknowledgments
We gratefully acknowledge the contributions of the following projects:  
1. https://github.com/IntelligentImaging/fetal-brain-extraction  
2. https://github.com/IvanDrokin/torch-conv-kan  
3. https://github.com/KindXiaoming/pykan  
4. https://github.com/gift-surg/NiftyMIC
