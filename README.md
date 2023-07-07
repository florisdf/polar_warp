# Polar Warp with PyTorch ❄️

The file `polar_warp.py` contains a `PolarWarp` PyTorch Module with which you can apply a polar warp transformation on PyTorch tensors. This allows you to do the transformation in batch and on GPU, which, according to our experiments, is up to 750 times faster than [OpenCV's implementation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga49481ab24fdaa0ffa4d3e63d14c0d5e4).

For more information, we refer to our paper [Rotation Equivariance for Diamond Identification](https://www.scitepress.org/Link.aspx?doi=10.5220/0011658400003417).
