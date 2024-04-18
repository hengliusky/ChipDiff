# ChipDiff

**This repository is for the paper, “Revisiting Chinese Ink Painting Style Transfer as Loss Gradient-Guided Staged Diffusion Sampling”**

Liu, H., Chen, C. . Revisiting Chinese Ink Painting Style Transfer as Loss Gradient-Guided Staged Diffusion Sampling

Given a landscape photo and a Chinese ink style painting, get an ink-wash stylized picture.



### Result

Result1:

![image-20240417104101499](https://github.com/hengliusky/ChipDiff/blob/main/README.assets/image-20240417104101499.png).

Result2:

![image-20240417104248358](https://github.com/hengliusky/ChipDiff/blob/main/README.assets/image-20240417104102327.png)



### Environment

Pytorch 1.9.0, Python 3.9

```
$ conda create --name lgss python=3.9
$ conda activate lgss
$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install color-matcher
```



### Download

To generate images, please download the pre-trained diffusion model[[baidu](https://pan.baidu.com/s/1Uo4tfodCfsHsdVqFEk_TkA?pwd=c8ii  "baidu")，[google_driver](https://drive.google.com/file/d/1dny31C7GEox50F10cDOwN1O6BwChrDqI/view?usp=drive_link "google_driver")] and put it into ```./checkpoints``` folder

Chinese ink painting dataset for training:[[baidu](https://pan.baidu.com/s/1hBiYPVd5Q2HVQVi0IWwLFQ?pwd=i6h1  "baidu")，[google_driver](https://drive.google.com/file/d/1x-JVuvGZohevxfhV23mbe_TTIfb6c-sa/view?usp=drive_link"google_driver")] 



### Testing

```
python3 main.py -i input_example/test3.8/content/hengshan1.png  --output_path outputs -tg input_example/test3.8/style/harvard_2.png --iterations_num 1   --use_range_restart  --use_noise_aug_all --use_colormatch
```



### Citing

If you use our work in your research, please cite us using the following BibTeX entry ~ Thank you .

```

```



