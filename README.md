# BCtypeFinder: Breast cancer subtype prediction framework based on the domain adaptation network with semi-supervised learning utilizing DNA methylation profiles

Breast cancer is a highly heterogeneous disease, leading to the varied drug resistance and clinical outcomes. Accurate identification of breast cancer subtypes is crucial for precise diagnosis, treatment decision-making, and prognosis prediction. Recent research has highlighted the significant role of epigenetic alterations in breast cancer development, particularly the potential of aberrant DNA methylation patterns as subtype-specific markers. However, challenges exist in developing a breast cancer subtype prediction model based on DNA methylation profiles, primarily due to the limited number of available samples with subtype information.

In this study, we propose BCtypeFinder, a breast cancer subtype prediction framework utilizing a domain adaptation network with semi-supervised learning. Our model leverages both labeled and unlabeled DNA methylation datasets to learn domain-invariant features, aligning the distributions of the same breast cancer subtypes across different datasets. BCtypeFinder outperforms existing methods, demonstrating superior classification performance in several scenarios. We also investigated the effectiveness of batch correction in BCtypeFinder, revealing its capability to eliminate batch distinctions among patients with the same subtype across different batches, thus enhancing the classifier's robustness.

## Requirements
* Python (>= 3.6)
* Pytorch (v1.6.0)

## Usage
Clone the repository or download source code files.

## Inputs
[Note!] All these inputs should be located in the cloned repository.
