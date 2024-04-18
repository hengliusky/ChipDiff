import torch
import torch.nn as nn
import torchvision.models as models

class vgg_loss(nn.Module):
    def __init__(self, content_layers=[4], style_layers=[1,2,3,4,5]):
        super(vgg_loss, self).__init__()
        #self.vgg = models.vgg19(pretrained=True).features.eval()
        self.vgg = models.vgg19(pretrained=True).features
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _get_features(self, x):
        features = []
        for name, module in self.vgg.named_children():
            name = int(name)
            x = module(x)
            if name in self.content_layers:
                features.append(x)
            elif name in self.style_layers:
                features.append(x)
        return features

    def get_content_features(self, x):
        x = self._normalize(x)
        con_list = []
        for each in self.content_layers:
            con_list.append(self._get_features(x)[each])
        return con_list

    def _gram_matrix(self, x):

        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        gram /= (b * c * h * w)

        return gram

    def get_style_features(self, x):
        x = self._normalize(x)
        features = self._get_features(x)
        style_features = []
        for feature in features:
            gram = self._gram_matrix(feature)
            #print(gram.shape)
            style_features.append(gram)
        return style_features

    def forward(self, x, content_target, style_target, content_weight=1, style_weight=100):
        x = self._normalize(x)
        x_con_features = self.get_content_features(x)
        x_sty_feature = self.get_style_features(x)
        content_loss = 0
        style_loss = 0
        for feature, target in zip(x_con_features, content_target):
            content_loss += torch.mean((feature - target)**2)
        for feature, target in zip(x_sty_feature, style_target):
            style_loss += torch.mean((feature - target)**2)


        return  content_loss, style_weight * style_loss
