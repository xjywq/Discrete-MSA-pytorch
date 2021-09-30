import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


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
        self.layer_parameters.append(layer.n_parameters())
        # self.layer_parameters.append([])
        # for name_param in layer.named_parameters():
        #     self.layer_parameters[self.n_layers-1].append(name_param)
        # for param_tensor, value in layer.state_dict().items():
        #     self.state_dict[param_tensor] = value

    def backward(self, loss):
        pout = torch.autograd.grad(loss, self.xs[-1])[0]
        for i in range(self.n_layers-1, -1, -1):
            space = {}
            parameters = self.layer_parameters[i]
            layer = self.layers[i]
            x = self.xs[i]
            for i in range(parameters):
                space[str(i)] = (-10, 10)
            optimizer = BayesianOptimization(
                f=None,
                pbounds=space,
                verbose=2,
                random_state=1)
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            for iter in range(10):
                rec = optimizer.suggest(utility)
                layer.set_var(rec)
                observation = layer.getH(x, pout)
                optimizer.register(params=rec, target=observation)
                # print('After %d iterations, best obj is %.3f' %
                #       (iter + 1, -opt.y.min()))

            # print(opt.X)
            # print(opt.y)
            best_x = optimizer.max['params']
            layer.set_var(best_x)
            # pout = layer.backward(x, pout)

    def parameters(self):
        params = []
        for i in range(self.n_layers):
            for named_param in self.layer_parameters[i]:
                params.append(named_param)
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


class FullyConnectLayer(AbstractLayer):
    def __init__(self, indim=64, outdim=64, dtype=torch.float64):
        super(FullyConnectLayer, self).__init__()
        self.indim, self.outdim = indim, outdim
        self.dtype = dtype
        self.model = nn.Linear(self.indim, self.outdim, dtype=dtype)

    def forward(self, input):
        return self.model(input)

    def n_parameters(self):
        return 12

    def set_var(self, rec):
        named_param = list(self.named_parameters())[0]
        theta = torch.tensor([rec[str(i)]
                             for i in range(9)], dtype=torch.float64).reshape(3, 3)
        named_param[1].requires_grad = False
        named_param[1].set_(theta)
        named_param[1].requires_grad = True
        named_param = list(self.named_parameters())[1]
        theta = torch.tensor([rec[str(i)]
                             for i in range(9, 12)], dtype=torch.float64).reshape(3)
        named_param[1].requires_grad = False
        named_param[1].set_(theta)
        named_param[1].requires_grad = True


if __name__ == '__main__':
    model = Model()
    model.add_layers(FullyConnectLayer(3, 3, torch.float64))
    model.add_layers(FullyConnectLayer(3, 3, torch.float64))
    # model.add_layers(Conv2d())
    x = torch.randn(3).to(torch.float64) * 10
    y = x ** 2 + np.random.uniform(0, 1, 3)
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # print(model.state_dict)
    print(x, y)
    for e in range(50):
        y_pred_1 = model(x)
        loss_1 = loss_fn(y_pred_1, y)
        print('epoch %d loss:' % e, loss_1.data)
        # model.zero_grad()
        # loss_1.backward()
        model.backward(loss_1)

    print(model(x), y)

    # model_normal = Conv2d_normal()
    # optimizer_norm = torch.optim.Adam(model_normal.parameters(), lr=lr)
    # for e in range(5):
    #     y_pred_2 = model_normal(x)
    #     loss_2 = loss_fn(y_pred_2, y)
    #     print('loss2:', loss_2)
    #     optimizer_norm.zero_grad()
    #     loss_2.backward()
    #     optimizer_norm.step()
