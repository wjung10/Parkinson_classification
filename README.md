# Parkinson_classification

This is the code repository of the manuscript *Development and Validation of a Deep Learning-based Automatic Brainstem Segmentation and Multi-class Classification Algorithm for Parkinsonian Syndromes using 3D T1-Weighted Images*.

## How to run

1. Open the terminal
2. Simply run the following code:

```shell
python run.py t1_path seg_path save_path
```

t1_path: input 3D T1 MRI image (must be saved in nii.gz format) \
seg_path: the associated brain parcellation mask containing brainstem region \
save_path: the saving path of the output brainstem segmentation mask containing midbrain, pons, medulla, and SCP

*Note* The full code (including DockerFile and an example case) would be uploaded after the acceptance.
