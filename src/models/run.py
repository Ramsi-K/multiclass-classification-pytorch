import argparse
import torch
from timeit import default_timer as timer

from TinyVGG import TinyVGG


# Set random seeds
torch.manual_seed(config[base][random_state])
torch.cuda.manual_seed(config[base][random_state])

# Set number of epochs
NUM_EPOCHS = config[train][epochs]
LR = config[train][lr]
device = config[base][device]

# Recreate an instance of TinyVGG
model_0 = TinyVGG(
    input_shape=3,  # number of color channels (3 for RGB)
    hidden_units=10,
    output_shape=len(train_data.classes),
).to(device)

# Setup loss function and optimizer
loss_fn = config[train][loss_fn]
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LR)

# Start the timer


start_time = timer()

# Train model_0
model_0_results = train(
    model=model_0,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
)

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-t", "--task", type=str, required=True, choices=["abnormal", "acl", "meniscus"]
#     )
#     parser.add_argument(
#         "-p",
#         "--plane",
#         type=str,
#         required=True,
#         choices=["sagittal", "coronal", "axial"],
#     )
#     parser.add_argument("--prefix_name", type=str, required=True)
#     parser.add_argument("--augment", type=int, choices=[0, 1], default=1)
#     parser.add_argument(
#         "--lr_scheduler", type=str, default="plateau", choices=["plateau", "step"]
#     )
#     parser.add_argument("--gamma", type=float, default=0.5)
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--lr", type=float, default=1e-5)
#     parser.add_argument("--flush_history", type=int, choices=[0, 1], default=0)
#     parser.add_argument("--save_model", type=int, choices=[0, 1], default=1)
#     parser.add_argument("--patience", type=int, default=5)
#     parser.add_argument("--log_every", type=int, default=100)
#     args = parser.parse_args()
#     return args
