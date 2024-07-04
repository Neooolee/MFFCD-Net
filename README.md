# MFFCD-Net
A Unified Cloud Detection Method for Suomi-NPP VIIRS Day and Night PAN Imagery  
# Abstract
Abstract—Cloud detection is a necessary step before the application of remote sensing images. However, the radiation intensity similarity between artificial lights and clouds is higher in nighttime remote sensing images than in daytime remote sensing images, making it difficult to distinguish artificial lights from clouds. This paper proposes a deep learning method called MFFCD-Net to detect clouds in daytime and nighttime remote sensing images. A dilated residual up-sampling module (DR-UP) for up-sampling feature maps while enlarging the receptive field. A multi-scale feature-extraction fusion module (MFEF) was designed to enhance the ability to distinguish regular textures of artificial lights from random textures of clouds. Moreover, an adaptive feature-fusion module (AFF) was designed to select and fuse the feature in the encoding stage and decoding stage, thus improving cloud detection accuracy. To the best of our knowledge, this is the first time a method is designed for cloud detection in both day and night time remote sensing images. The experimental results on Suomi-NPP Visible Infrared Imaging Radiometer Suite (VIIRS) of the panchromatic (PAN) day/night band (DNB) images show that MFFCD-Net could obtain a better balance in commission and omission rates than baseline methods (92.3% versus 90.5% on the F1-score) in daytime remote sensing images. Although artificial lights introduce strong interference in nighttime remote sensing images, MFFCD-Net can better distinguish artificial lights from clouds than baseline methods (90.8% versus 88.4% on the F1-score). The results indicates that MFFCD-Net is promising for cloud detection both in daytime and nighttime remote sensing images. The source code and dataset are available on https://github.com/Neooolee/MFFCD-Net.

The dataset used in this work is avaliable on:...
