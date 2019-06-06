import schnetpack.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *

# load qm9 dataset and download if necessary
data = MD17("md17.db", molecule='benzene', properties=[MD17.energy], collect_triples=True)

# split in train and val
train, val, test = data.create_splits(40000, 1000)
loader = spk.data.AtomsLoader(train, batch_size=64, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
reps = rep.BehlerSFBlock()
output = schnetpack.output_modules.ElementalAtomwise(reps.n_symfuncs)
model = atm.AtomisticModel(reps, output)

# filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())

# create trainer
opt = Adam(trainable_params, lr=1e-4)
#loss = lambda b, p: F.mse_loss(p["y"], b[MD17.energy])
def loss(b, p):
    return F.mse_loss(p["y"], b[MD17.energy]+145484.55693482968)
trainer = spk.train.Trainer("wacsf/", model, loss, opt, loader, val_loader)
trainer.mae_loss_fn = lambda b,p: F.l1_loss(p["y"], b[MD17.energy]+145484.55693482968)

# start training
trainer.train(torch.device("cuda"))
