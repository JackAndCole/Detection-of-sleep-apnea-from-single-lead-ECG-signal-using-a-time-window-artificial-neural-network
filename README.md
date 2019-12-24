#  Code for: Detection of sleep apnea from single-lead ECG signal using a time window artificial neural network

## Abstract

Sleep apnea (SA) is a ubiquitous sleep-related respiratory disease. It can occur hundreds of times at night, and its long-term occurrences can lead to some serious cardiovascular and neurological diseases. Polysomnography (PSG) is a commonly used diagnostic device for SA. But it requires suspected patients to sleep in the lab for one to two nights and records about 16 signals through expert monitoring. The complex processes hinder the widespread implementation of PSG in public health applications. Recently, some researchers have proposed using a single-lead ECG signal for SA detection. These methods are based on the hypothesis that the SA relies only on the current ECG signal segment. However, SA has time dependence; that is, the SA of the ECG segment at the previous moment has an impact on the current SA diagnosis. In this study, we develop a time window artificial neural network that can take advantage of the time dependence between ECG signal segments and does not require any prior assumptions about the distribution of training data. By verifying on a real ECG signal dataset, the performance of our method has been significantly improved compared to traditional non-time window machine learning methods as well as previous works.

# Dataset

[apnea-ecg](https://physionet.org/content/apnea-ecg/1.0.0/), [event-1-answers](dataset/event-1-answers), [event-2-answers](dataset/event-2-answers)

## Cite

If our work is helpful to you, please cite:

Tao Wang, Changhua Lu, and Guohao Shen, “Detection of Sleep Apnea from Single-Lead ECG Signal Using a Time Window Artificial Neural Network,” BioMed Research International, vol. 2019, Article ID 9768072, 9 pages, 2019. [https://doi.org/10.1155/2019/9768072](https://doi.org/10.1155/2019/9768072).

## Email:

If you have any questions, please email to: wtustc@mail.ustc.edu.cn
