import numpy as np
import torch
import torch.nn as nn
import jax
import equinox as eqx

# load saved equinox model
LOAD_MODEL_NAME = "./checkpoints/RBC_1e4_noisy_data.eqx"
SAVE_MODEL_NAME = "./checkpoints/traced_rbc_model_1e4_noisy_data.pt"
units = 100
# units = 50 # for few data case
pinn = eqx.nn.MLP(
    in_size=2, 
    out_size=1, 
    width_size=units, 
    depth=4,
    key=jax.random.PRNGKey(42)
)
pinn = eqx.tree_deserialise_leaves(LOAD_MODEL_NAME, pinn)
print("Number of layers of the Equinox model: ", len(pinn.layers))


## build PyTorch model
class NeuralNetwork(nn.Module):
    
    def __init__(self, units=50):
        super().__init__()
        self.linear1 = nn.Linear(2, units)
        self.linear2 = nn.Linear(units, units)
        self.linear3 = nn.Linear(units, units)
        self.linear4 = nn.Linear(units, units)
        self.linear5 = nn.Linear(units, 1)
        
    def forward(self, x, y):
        out = torch.concat([x, y], dim=-1)
        out = torch.tanh(self.linear1(out))
        out = torch.tanh(self.linear2(out))
        out = torch.tanh(self.linear3(out))
        out = torch.tanh(self.linear4(out))
        return self.linear5(out)
    

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

## save PyTorch model

x = np.linspace(0, 1, 2)
y = np.linspace(0, 1, 2)
x_grid, y_grid = np.meshgrid(x, y)
xx = torch.tensor(x_grid.reshape([-1, 1]), dtype=torch.float32)
yy = torch.tensor(y_grid.reshape([-1, 1]), dtype=torch.float32)
example = torch.cat((xx,yy), dim=1)
example = [xx, yy] 
out = model.forward(xx, yy)

print(f"out: {out}")
print(f"Exampe: {example}")

print(f"Example : {example}")
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(SAVE_MODEL_NAME)

#model = torch.jit.script(model)
#
#model.save("./model.pt")
