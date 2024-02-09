from torchvision import datasets, transforms
from torch.utils.data import DataLoader


train_dir = "..\\..\\data\\processed\\train"
val_dir = "..\\..\\data\\processed\\valid"
test_dir = "..\\..\\data\\processed\\test"

data_transform = transforms.Compose(
    [transforms.Resize(size=(64, 64)), transforms.ToTensor()]  # and normalize
)


train_data = datasets.ImageFolder(
    root=train_dir, transform=data_transform, target_transform=None
)
val_data = datasets.ImageFolder(root=val_dir, transform=data_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

print(
    f"Train data:\n{train_data}\
      \nVal data:\n{val_data}\
      \nTest data:\n{test_data}"
)

# Import data from directories and turn it into batches
train_dataloader = DataLoader(
    dataset=train_data, batch_size=32, num_workers=1, shuffle=True
)
val_dataloader = DataLoader(
    dataset=val_data, batch_size=32, num_workers=1, shuffle=False
)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=32, num_workers=1, shuffle=False
)

print(
    f"train_dataloader: {train_dataloader}\
      val_dataloader: {val_dataloader}\
        test_dataloader: {test_dataloader}"
)

img, label = next(iter(train_dataloader))

print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")
