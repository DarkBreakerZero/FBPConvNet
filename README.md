# FBPConvNet

An unofficial version of FBPConvNet. Some details are different from the original paper.

1. Install torch-randon (https://github.com/matteo-ronchetti/torch-radon)
2. Generate simulated sparse-view CT data: gen_ld_data.py
3. Generate training data: make_proj_img_list.py
4. Train and Validation: train_fbpconvnet.py
5. Test: test_fbpconvnet.py

Please cite the following references:

1. TorchRadon: Fast Differentiable Routines for Computed Tomography
2. Deep convolutional neural network for inverse problems in imaging
3. DREAM-Net: Deep Residual Error iterAtive Minimization Network for Sparse-View CT Reconstruction
