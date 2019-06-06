import schnetpack.output_modules
import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *

# load qm9 dataset and download if necessary
data = QM9("qm9.db", properties=[QM9.U0])

# split in train and val
train, val, test = data.create_splits(10, 10)
loader = spk.data.AtomsLoader(train, batch_size=100, num_workers=4)
val_loader = spk.data.AtomsLoader(val)

# create model
reps = rep.SchNet()
output = schnetpack.output_modules.Atomwise()
model = atm.AtomisticModel(reps, output)

# create trainergit add
opt = Adam(model.parameters(), lr=1e-4)
loss = lambda b, p: F.mse_loss(p["y"], b[QM9.U0])
def mae_loss_fn(b, p):
    return F.l1_loss(p["y"], b[QM9.U0])
loss.mae_loss_fn = mae_loss_fn
trainer = spk.train.Trainer("output/", model, loss, opt, loader, val_loader)

# start training
trainer.train(torch.device("cpu"))
#trainer.train(torch.device("cuda"))
