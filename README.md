# App:  Brain Segmentation

Deep learning app made for T1-weighted MRI brain segmentation using ANTsRNet

## Model training notes

* Training data: IXI, NKI, Kirby, Oasis, ADNI SSTs
* Unet model (see ``Scripts/Training/``).
* Template-based data augmentation

## Sample prediction usage

```
#
#  Usage:
#    Rscript doBrainExtraction.R inputImage inputImageBrainExtractionMask outputImage reorientationTemplate
#
#  MacBook Pro 2016 (no GPU)
#

$ Rscript Scripts/doBrainTissueSegmentation.R Data/Example/1097782_defaced_MPRAGE.nii.gz Data/Example/1097782_defaced_MPRAGEBrainExtractionMask.nii.gz ../output Data/Template/S_template3_resampled2.nii.gz

Reading reorientation template Data/Template/S_template3_resampled2.nii.gz  (elapsed time: 0.146404 seconds)
Using TensorFlow backend.
Loading weights file2018-12-11 18:12:13.558871: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
  (elapsed time: 0.308912 seconds)
Reading  Data/Example/1097782_defaced_MPRAGE.nii.gz  (elapsed time: 0.404892 seconds)
Normalizing to template and cropping to mask.  (elapsed time: 0.7775259 seconds)
Prediction and decoding (elapsed time: 22.09039 seconds)
Renormalize to native space  (elapsed time: 3.351701 seconds)
Writing ../output  (elapsed time: 4.786408 seconds)

Total elapsed time: 31.67444 seconds
```

## Sample results

![Brain extraction results](Documentation/Images/resultsBrainExtraction.png)
