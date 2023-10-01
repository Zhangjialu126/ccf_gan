# Neural Characteristic Function Learning for Conditional Image Generation (ICCV 2023)
This code is for the implementation of CCF-GAN proposed in the paper "[Neural Characteristic Function Learning for Conditional Image Generation](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Neural_Characteristic_Function_Learning_for_Conditional_Image_Generation_ICCV_2023_paper.pdf)" that has been accepted in ICCV 2023.

[[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Neural_Characteristic_Function_Learning_for_Conditional_Image_Generation_ICCV_2023_paper.pdf) [[supp]](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Li_Neural_Characteristic_Function_ICCV_2023_supplemental.pdf) [[poster]](/assets/poster.pdf)

## Requirements
### For BigGAN platform:
- Python 3.7
- PyTorch
- ```pip install -r requirements_biggan.txt```
### For StudioGAN platform:
- Python 3.7
- PyTorch (at least 1.7)
- ```pip install -r requirements_studiogan.txt```

## Citation
CCF-GAN implementation is heavily based on [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) and [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN). If you use this code, please cite

```
@InProceedings{Li_2023_ICCV,
    author    = {Li, Shengxi and Zhang, Jialu and Li, Yifei and Xu, Mai and Deng, Xin and Li, Li},
    title     = {Neural Characteristic Function Learning for Conditional Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7204-7214}
}
```
