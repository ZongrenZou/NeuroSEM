import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import jax
import equinox as eqx


LOAD_MODEL_NAME = "./checkpoints/PINN2.eqx"
SAVE_MODEL_NAME = "./checkpoints/traced_pinn2.pt"
units = 100
pinn = eqx.nn.MLP(
    in_size=3,
    out_size=3,
    width_size=units,
    depth=5,
    activation=jax.numpy.tanh,
    key=jax.random.PRNGKey(99999),
)
pinn = eqx.tree_deserialise_leaves(LOAD_MODEL_NAME, pinn)
print("Number of layers of the Equinox model: ", len(pinn.layers))


## build PyTorch model
class NeuralNetwork(nn.Module):
    
    def __init__(self, units=50):
        super().__init__()
        self.linear1 = nn.Linear(3, units)
        self.linear2 = nn.Linear(units, units)
        self.linear3 = nn.Linear(units, units)
        self.linear4 = nn.Linear(units, units)
        self.linear5 = nn.Linear(units, units)
        self.linear6 = nn.Linear(units, 1)

        # self.activation = torch.sin
        self.activation = torch.tanh
        
    def forward(self, x, y, t):
        out = torch.concat([x, y, t], dim=-1)
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out = self.activation(self.linear3(out))
        out = self.activation(self.linear4(out))
        out = self.activation(self.linear5(out))
        return self.linear6(out)


model = NeuralNetwork(units=units)

# copy and paste
# TODO: write it in for-loop
model.linear1.weight = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[0].weight)
    ), requires_grad=False,
)
model.linear1.bias = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[0].bias)
    ), requires_grad=False,
)
model.linear2.weight = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[1].weight)
    ), requires_grad=False,
)
model.linear2.bias = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[1].bias)
    ), requires_grad=False,
)
model.linear3.weight = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[2].weight)
    ), requires_grad=False,
)
model.linear3.bias = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[2].bias)
    ), requires_grad=False,
)
model.linear4.weight = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[3].weight)
    ), requires_grad=False,
)
model.linear4.bias = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[3].bias)
    ), requires_grad=False,
)
model.linear5.weight = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[4].weight)
    ), requires_grad=False,
)
model.linear5.bias = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[4].bias)
    ), requires_grad=False,
)
model.linear6.weight = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[5].weight)
    ), requires_grad=False,
)
model.linear6.bias = nn.parameter.Parameter(
    torch.tensor(
        np.array(pinn.layers[5].bias)
    ), requires_grad=False,
)


## save PyTorch model
xyt = sio.loadmat("./data/gauss_quad.mat")["xyt"]
x, y, t = np.split(xyt, 3, axis=-1)
# xy = np.concatenate([x, y], axis=-1)[..., None]


# print(x)
# print("############################")
# print(y)
# print("############################")
# print(t)
# print("############################")

# x = np.random.normal(size=[7, 1])
# y = np.random.normal(size=[7, 1])
# t = np.random.normal(size=[7, 1])

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.float32)

example = torch.cat((x, y, t), dim=1)
example = [x, y, t] 
out = model.forward(x, y, t)

print(f"out: {out}")
# print(f"Exampe: {example}")

# print(f"Example : {example}")
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(SAVE_MODEL_NAME)

sio.savemat("./data/output2.mat", {"xyt": xyt, "uvp": out})

#model = torch.jit.script(model)
#
#model.save("./model.pt")