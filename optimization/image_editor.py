import os

from utils_visualize.metrics_accumulator import MetricsAccumulator

import imageio
from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image

from torchvision import transforms

from torchvision.transforms import functional as TF

from optimization.losses import range_loss

import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit

from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
from pathlib import Path
from id_loss import IDLoss

from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer
from feature_loss.loss import mean_std_loss, to_gray
from feature_loss.fea_loss import vgg_loss
from util_hed.hed import hed_loss
from util_hed.color_hist import hist_loss
import re

mean_sig = lambda x:sum(x)/len(x)
class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(self.args.output_path)
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        if self.args.use_ffhq:
            self.model_config.update(
            {
                "attention_resolutions": "16",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 128,
                "num_head_channels": 64,
                "num_res_blocks": 1,
                "resblock_updown": True,
                "use_fp16": False,
                "use_scale_shift_norm": True,
            }
        )
        else:
            self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": False,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        

        self.model.load_state_dict(
            torch.load(
                "./checkpoints/ckpt_model.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        with open("model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config

        #一次尝试，小于最后50时，仅保留内容损失


        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=args.lambda_trg).eval()#.requires_grad_(False)

        self.fea_loss = vgg_loss().cuda()



        
        self.cm = ColorMatcher()
        self.clip_size = 224
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
#         self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()


    def Hed_loss(self, a, b, t, p, show):
        return hed_loss(a, b, t, p, show)


    def noisy_aug(self,t,x,x_hat):
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def edit_image(self):

        #加载图片
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        #512*512


        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        
        self.target_image = None
        if self.args.target_image is not None:
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (
                TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )

        con_img = ''
        match = re.search(r'\/(\w+)\.png', self.args.init_image)
        if match:
            con_img = match.group(1)

        sty_img = ''
        match = re.search(r'\/(\w+)\.png', self.args.target_image)
        if match:
            sty_img = match.group(1)
        print('testing: content {0}, style {1}'.format(con_img, sty_img))
        print('target.std:', torch.std(self.target_image))


        self.prev = self.init_image.detach()

        self.flag_resample=False
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1


        #内函数，输入图片与步长，计算各种损失，得到梯度以及重采样控制符
        def cond_fn(x, t, y=None):

            self.flag_resample=False
            with torch.enable_grad():
                frac_cont=1.0
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                
                loss = torch.tensor(0)

                if self.args.use_noise_aug_all:
                    x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                else:
                    x_in = out["pred_xstart"]

                if self.args.vit_lambda != 0:
                    if t[0]>self.args.diff_iter:
                        vit_loss,vit_loss_val, gobal_ssim_loss, cls_style_loss = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                    else:

                        vit_loss,vit_loss_val, gobal_ssim_loss, cls_style_loss = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                    loss = loss + vit_loss


                    loss += gobal_ssim_loss
                    if t[0] < self.args.style_step:#默认40，挺好的
                        loss += cls_style_loss


                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                if self.target_image is not None :
                    color_loss = (hist_loss(x_in, self.target_image, self.args.cor_mode)* self.args.l2_hist + mean_std_loss( x_in, self.target_image, self.args.cor_mode)*5) * self.args.l2_trg_lambda
                    loss = loss + color_loss

                '''加入基于feature特征图的内容与风格损失'''

                pred_image = out["pred_xstart"]

                stylized = pred_image.add(1).div(2).clamp(0, 1)
                content_features = self.fea_loss.get_content_features(self.init_image)
                style_features = self.fea_loss.get_style_features(self.target_image)#由特征图计算得到的gram矩阵
                content_loss, style_loss = self.fea_loss(stylized, content_features, style_features)

                if t[0] < self.args.style_step:
                    loss += style_loss*self.args.lambda_sty_fea
                loss += content_loss * self.args.lambda_con_fea


                #新增hedloss

                if self.args.lambda_hed > 0:
                    hed_loss = self.Hed_loss(pred_image, self.init_image, t, str(self.ranked_results_path)+'/{}_{}_hed{}'.format(con_img, sty_img, iteration_number), self.args.show_hed)*self.args.lambda_hed
                    loss += hed_loss



                self.prev = x_in.detach().clone()
                
                if self.args.use_range_restart:
                    if t[0].item() < total_steps:
                        if self.args.use_ffhq:
                            if r_loss>2.9:#0.3
                                self.flag_resample =True
                        else:
                            if r_loss>2.9:
                                self.flag_resample =True

            #print('总损失',loss)
            return -torch.autograd.grad(loss, x)[0], self.flag_resample

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")
    
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None,
                randomize_class=True,
                use_inversion=self.args.use_inversion
            )
            if self.flag_resample:
                continue


            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 
            total_steps_with_resample = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (self.args.resample_num-1)
            if_save = False
            for j, sample in enumerate(samples):
                # if sample['time'] > 10:
                #     continue
                if_save = True
                should_save_image = j % save_image_interval == 0 or j == total_steps_with_resample
                # self.metrics_accumulator.print_average_metric()
                for b in range(self.args.batch_size):
                    pred_image = sample["pred_xstart"][b]
                    # visualization_path = visualization_path.with_name(
                    #     f"{iteration_number}_{self.args.lambda_contra_ssim}_{self.args.lambda_dir_cls}_{self.args.lambda_ssim}_{self.args.vit_lambda}_{self.args.range_lambda}_{self.args.l2_trg_lambda}_{self.args.lambda_sty_fea}_{self.args.lambda_con_fea}_{self.args.lambda_trg}_{self.args.lambda_hed}{visualization_path.suffix}"
                    # )
                    pred_image = pred_image.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)

            if if_save == False:
                print('生成失败')
                continue

            visualization_path = Path(
                os.path.join(self.args.output_path, self.args.output_file)
            )

            visualization_path = visualization_path.with_name(
                f"{iteration_number}_{con_img}_stylized_{sty_img}{visualization_path.suffix}"
            )
            ranked_pred_path = self.ranked_results_path / (visualization_path.name)




            if self.args.target_image is not None:
                #print(len(sample["out_list"]))
                if self.args.use_colormatch:
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, str(ranked_pred_path))
                    #pred_image_pil.save(str(ranked_pred_path)[:-4]+'_ori.png')
                else:
                    print('保存')
                    pred_image_pil.save(str(ranked_pred_path))


            if self.args.show_center:
                save_dir = str(self.ranked_results_path/'{}_{}_center{}'.format(con_img, sty_img, iteration_number))
                if os.path.exists(save_dir) == 0:
                    os.makedirs(save_dir)

                frame_x0 = []
                frame_xt = []
                for i in range(len(sample["out_list"])):#out_list: x0， xts：xt
                    pic = sample["out_list"][i][0]
                    pred_image = pic.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, save_dir+'/x0_output_{}.png'.format(i))
                    #self.add_title(save_dir+'/output_{}.png'.format(i), str(i))
                    frame_x0.append(imageio.imread(save_dir+'/x0_output_{}.png'.format(i)))

                imageio.mimsave((save_dir+'/Gif-x0.gif'), frame_x0)
                for i in range(len(sample["xts"])):#out_list: x0， xts：xt
                    pic = sample["xts"][i][0]
                    pred_image = pic.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, save_dir+'/xt_output_{}.png'.format(i))
                    #self.add_title(save_dir+'/output_{}.png'.format(i), str(i))
                    frame_xt.append(imageio.imread(save_dir+'/xt_output_{}.png'.format(i)))

                imageio.mimsave((save_dir+'/xt_Gif-x0.gif'), frame_xt)
