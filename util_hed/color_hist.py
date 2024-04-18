import torch,math
import torch.nn.functional as F
def fun(a,b,x):
    y = torch.pow(x,2)-(a+b)*x+a*b
    return 1/(1+torch.pow(math.e, y*5))



def get_color_vetor(x,n_class = 20,max = 1.0,min = -1.0):
    x = x.view(3,-1)
    step = (max-min)/n_class
    color_vetor = fun(min, min+step, x).sum(1).unsqueeze(0)
    for i in range(1, n_class):
        color_vetor = torch.cat((color_vetor,fun(min+i*step, min+(i+1)*step, x).sum(1).unsqueeze(0)), 0)
    #将【1，3】的统计向量转为1维颜色连续分布向量
    color_vetor = color_vetor.view(-1)
    min = torch.min(color_vetor)
    max = torch.max(color_vetor)
    normlized_vetor = (color_vetor-min)/(max-min)

    return normlized_vetor

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

def hist_loss(a, b, mode='rgb'):
    if mode == 'lab':
        #return 0
        a = to_lab(a)
        b = to_lab(b)
    l = F.mse_loss(get_color_vetor(b),get_color_vetor(a))

    return l



