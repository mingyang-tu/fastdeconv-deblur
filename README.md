# Deblurring 

Python implementation of Krishnan et al. "Fast image deconvolution using hyper-Laplacian priors." (2009)

## Requirements

- numpy
- scipy
- opencv-python

## Usage

See `example.py`

```python
from fastdeconv import FastDeconvolution

fd = FastDeconvolution(blurred_image, kernel, lambda_, alpha)
deblurred = fd.solve()
```
