# BCtypeFinder: Breast cancer subtype prediction framework based on the domain adaptation network with semi-supervised learning utilizing DNA methylation profiles

Breast cancer is a highly heterogeneous disease, leading to the varied drug resistance and clinical outcomes. Accurate identification of breast cancer subtypes is crucial for precise diagnosis, treatment decision-making, and prognosis prediction. Recent research has highlighted the significant role of epigenetic alterations in breast cancer development, particularly the potential of aberrant DNA methylation patterns as subtype-specific markers. However, challenges exist in developing a breast cancer subtype prediction model based on DNA methylation profiles, primarily due to the limited number of available samples with subtype information.

In this study, we propose BCtypeFinder, a breast cancer subtype prediction framework utilizing a domain adaptation network with semi-supervised learning. Our model leverages both labeled and unlabeled DNA methylation datasets to learn domain-invariant features, aligning the distributions of the same breast cancer subtypes across different datasets. BCtypeFinder outperforms existing methods, demonstrating superior classification performance in several scenarios. We also investigated the effectiveness of batch correction in BCtypeFinder, revealing its capability to eliminate batch distinctions among patients with the same subtype across different batches, thus enhancing the classifier's robustness.

## Requirements
* Python (>= 3.6)
* Pytorch (v1.6.0)

## Usage
Clone the repository or download source code files.

## Inputs
[Note!] All the example datasets can be found in './dataset/' directory.

### 1. Source dataset
* Source_X
  - Contains the methylation profiles for the source dataset
  - Row : Sample, Column : Feature (CpG)
  - The first column should be the "sample_id" and the first row should contain the feature names.
  - Example : ./dataset/source_X.csv
* Source_Y
  - Contains the integer-converted subtype information for the source dataset
  - The first row should contain the "sample_id" and "subtype" column names. The sample_id should be sorted in the same way as the ones in **Source_X**.
  - Example : ./dataset/source_Y.csv

### 2. Target dataset
* Target_X
   - Contains the methylation profiles for the labeled dataset
   - Row : Sample, Column : Feature (CpG)
   - The first column should be the "sample_id" and the last coulmn shoud be "domain_idx" which contains the integer number (index) discriminating each dataset. Samples in the same dataset should have same number.
   - The first row should contain the feature names.
   - Example : ./dataset/target_X.csv

### 3. Test dataset
* Contains the testing dataset to evaluate BCtypeFinder
* The first column should be the "sample_id" and the last coulmn shoud be "subtype" which contains the integer-converted subtype label for each sample.
* The first row should contain the feature names.
* Example : ./dataset/target_test.csv

## How to run
1. Edit the **run_BCtypeFinder.sh** to make sure each variable indicate the corresponding source, target and test dataset files as input.
2. Run the below command :
```
chmod +x run_BCtypeFinder.sh
./run_BCtypeFinder.sh
```
3. All the results will be saved in the newly created **results** directory.
   * ft_test_target_pred.csv : predicted subtype label for testing dataset
   * ft_test_target_label.csv : actual subtype label of testing dataset you provided for evaluation

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.
