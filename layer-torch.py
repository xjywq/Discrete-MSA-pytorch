import numpy as np
import torch
import torch.nn as nn
from hebo.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.n_layers = 0
        self.layers = []
        self.layer_parameters = []
        self.xs = []
        self.out = 0
        # self.state_dict = {}
        # self.named_parameters = []

    def forward(self, x):
        self.xs.append(x)
        for layer in self.layers:
            x = layer.forward(x)
            self.xs.append(x)
        return self.xs[-1]

    def add_layers(self, layer):
        self.layers.append(layer)
        self.n_layers += 1
        self.layer_parameters.append(layer.named_parameters())
        # for name_param in layer.named_parameters():
        #     self.named_parameters.append((name_param))
        # for param_tensor, value in layer.state_dict().items():
        #     self.state_dict[param_tensor] = value

    def backward(self, loss):
        pout = torch.autograd.grad(loss, self.xs[-1])[0]
        for i in range(self.n_layers-1, -1, -1):
            space = []
            parameters = self.layer_parameters[i]
            layer = self.layers[i]
            x = self.xs[i]
            for named_param in parameters:
                print(named_param[0])
                space.append({'name': named_param[0],
                              'type': 'int', 'lb': -1, 'ub': 1})
            space = DesignSpace().parse(space)
            opt = HEBO(space)
            for iter in range(50):
                rec = opt.suggest()
                for named_param in parameters:
                    named_param.data.set_(
                        rec.iloc[0].__getattr__(named_param[0]))
                observation = -layer.getH(x, pout)
                opt.observe(rec, np.array([observation]))
                print('After %d iterations, best obj is %.3f' %
                      (iter + 1, opt.y.min()))
            for named_param in parameters:
                named_param.data.set_(
                    opt.best_x.iloc[0].__getattr__(named_param.name))
            pout = layer.backward(x, pout)

    def parameters(self):
        params = []
        for i in range(self.n_layers):
            for named_param in self.layer_parameters[i]:
                params.append(named_param)
                print(named_param)
        return params


class AbstractLayer(torch.nn.Module):
    def __init__(self):
        super(AbstractLayer, self).__init__()
        self.vars = []

    def forward(self):
        return

    def backward(self, x, p):
        H = torch.sum(p * self.forward(x))
        p_next = torch.autograd.grad(H, x)[0]
        return p_next

    def getH(self, x, p):
        return torch.sum(torch.mul(p, self.forward(x)))



class Conv2d(AbstractLayer):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, stride=[1, 1], activation_fn=True, bn=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_fn = activation_fn
        self.bn = bn

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)
        if bn:
            self.use_bn = torch.nn.BatchNorm2d(out_channels)
        if activation_fn:
            self.activation_fn = torch.nn.ReLU(inplace=True)

    def forward(self, input):
        # kernel_h, kernel_w = self.kernel_size
        # kernel = _variable_with_weight_decay('weights',shape=kernel_shape,use_xavier=use_xavier,stddev=stddev,wd=weight_decay)
        input = input.permute(0, 3, 2, 1)
        input = self.conv(input)
        # if self.bn:
        #     input = self.use_bn(input)
        # if self.activation_fn:
        #     input = self.activation_fn(input)
        output = input.permute(0, 3, 2, 1)    # B N 3
        return output


class Conv2d_normal(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, stride=[1, 1], activation_fn=True, bn=False):
        super(Conv2d_normal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_fn = activation_fn
        self.bn = bn

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

        if bn:
            self.use_bn = torch.nn.BatchNorm2d(out_channels)

        if activation_fn:
            self.activation_fn = nn.ReLU(inplace=True)

    def forward(self, input):
        # kernel_h, kernel_w = self.kernel_size
        # kernel = _variable_with_weight_decay('weights',shape=kernel_shape,use_xavier=use_xavier,stddev=stddev,wd=weight_decay)
        input = input.permute(0, 3, 2, 1)
        input = self.conv(input)
        if self.bn:
            input = self.use_bn(input)
        if self.activation_fn:
            input = self.activation_fn(input)
        output = input.permute(0, 3, 2, 1)
        return output


if __name__ == '__main__':
    model = Model()
    model.add_layers(Conv2d())
    # model.add_layers(Conv2d())
    x = torch.randn(1, 5, 5, 3)
    y = torch.randn(1, 3, 3, 16)
    loss_fn = nn.MSELoss()
    lr = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # print(model.state_dict)
    for e in range(5):
        y_pred_1 = model(x)
        loss_1 = loss_fn(y_pred_1, y)
        print('loss1:', loss_1)
        # model.zero_grad()
        # loss_1.backward()
        print(loss_1)
        model.backward(loss_1)

    # model_normal = Conv2d_normal()
    # optimizer_norm = torch.optim.Adam(model_normal.parameters(), lr=lr)
    # for e in range(5):
    #     y_pred_2 = model_normal(x)
    #     loss_2 = loss_fn(y_pred_2, y)
    #     print('loss2:', loss_2)
    #     optimizer_norm.zero_grad()
    #     loss_2.backward()
    #     optimizer_norm.step()
