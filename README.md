# msa-arxiv2020
This repository hosts the sources to reproduce the following study: A Multisensory Learning Architecture for Rotation-invariantObject Recognition [1]. The employed dataset can be reached via [http://www.robotmultimodal.com/datasets/](http://www.robotmultimodal.com/datasets/). The dataset report can be found in [2].  


> **Abstract:** This study presents a multisensory machine learning architecture for object recognition by employing a novel dataset that was constructed with the iCub robot, which is equipped with three cameras and a depth sensor. The proposed architecture combines convolutional neural networks to form representations (i.e., features) for grayscaled color images and a multi-layer perceptron algorithm to process depth data. To this end, we aimed to learn joint representations of different modalities (e.g., color and depth) and employ them for recognizing objects. We evaluate the performance of the proposed architecture by benchmarking the results obtained with the models trained separately with the input of different sensors and a state-of-the-art data fusion technique, namely decision level fusion. The results show that our architecture improves the recognition accuracy compared with the models that use inputs from a single modality and decision level multimodal fusion method.  

## Folders 
+ **models_h5:** contains the trained models' parameter in h5 format.  
+ **recognition_io_pkl:** contains the input matrices (train, val, test) in .pkl format for each sensors.  
+ **results_pkl:** contains the prediction, loss and accuracy trace, and related vectors for generate confusion matrices. Note that the **y_testing.pkl** is true label of the objects in test set.  
+ **visualization:** contains the intermediate fusion archictecture diagram.  
+ **src:** contains source code.  Since the models for icub's right and left cameras are the same model with **realsense_cnn**, you can run this file by chaning the line below:  

```python
   Xtr, Xval, Xtst = hlp.load_rsense_inputs() # 2-change sensory input to _iright_ or _ileft_
```

## References
[1] Kirtay, M., Schillaci, G., & Hafner, V. V. (2020). A Multisensory Learning Architecture for Rotation-invariant Object Recognition. arXiv preprint arXiv:2009.06292.  
[2] Kirtay, M., Albanese, U., Vannucci, L., Schillaci, G., Laschi, C., & Falotico, E. (2020). The iCub multisensor datasets for robot and computer vision applications. arXiv preprint arXiv:2003.01994.  

