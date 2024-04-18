import torch
import torch.nn.functional as F


def mean_std_loss(input, target, mode = 'rgb'):
    if mode == 'lab':
        input = to_lab(input)/200
        target = to_lab(target)/200
    mean_tensor1 = torch.mean(input)
    mean_tensor2 = torch.mean(target)
    std_tensor1 = torch.std(input)
    std_tensor2 = torch.std(target)
    mul = mean_tensor2/std_tensor2
    #print('颜色损失中，均值：{0}； 方差：{1}'.format(mean_tensor2, std_tensor2))
    #print('颜色损失中，均值：{0}； 方差：{1}'.format(mean_tensor1, std_tensor1))

    return ((mean_tensor2-mean_tensor1)**2)+((std_tensor2*mul-std_tensor1*mul)**2)

def xyz_to_lab(xyz):#xyz:[1,3,256,256]
    xyz_ref = torch.tensor([95.047, 100.000, 108.883]).view(1, 3, 1, 1).cuda()  # D50 Illuminant的白点
    xyz_normalized = xyz / xyz_ref

    mask = xyz_normalized > 0.008856
    xyz_normalized[mask] = torch.pow(xyz_normalized[mask], 1 / 3)
    xyz_normalized[~mask] = (xyz_normalized[~mask] * 903.3 + 16) / 116

    lab = torch.zeros_like(xyz_normalized)
    lab[:, 0, :, :] = torch.clamp(116 * xyz_normalized[:, 1, :, :] - 16, min=0)
    lab[:, 1, :, :] = (xyz_normalized[:, 0, :, :] - xyz_normalized[:, 1, :, :]) * 500
    lab[:, 2, :, :] = (xyz_normalized[:, 1, :, :] - xyz_normalized[:, 2, :, :]) * 200

    return lab

def to_lab(rgb_image_tensor):
    rgb_image_normalized = (rgb_image_tensor + 1) * 0.5
    rgb_to_xyz_matrix = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                      [0.2126729, 0.7151522, 0.0721750],
                                      [0.0193339, 0.1191920, 0.9503041]]).cuda()
    # 将RGB图像数据标准化到[0, 1]范围
    # 转换RGB到XYZ颜色空间
    rgb_image_normalized = rgb_image_normalized*255
    rgb_image_normalized = rgb_image_normalized.permute(0, 2, 3, 1)
    xyz_image = torch.matmul(rgb_image_normalized, rgb_to_xyz_matrix.T)
    xyz_image = xyz_image.permute(0, 3, 1, 2)

    lab_image = xyz_to_lab(xyz_image)
    return lab_image

def to_gray(img):

    # 将RGB三通道的像素值取平均，得到灰度值
    gray_img = torch.mean(img, dim=1, keepdim=True)
    # 将灰度Tensor复制为3通道Tensor，得到黑白图片
    bw_img = torch.cat([gray_img, gray_img, gray_img], dim=1)
    return bw_img

