import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as T
import random
from tqdm import tqdm
from torchsummary import summary


cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# Custom Cutout transform
class RandomCutout(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        mask_size_half = self.mask_size // 2

        cx = random.randint(mask_size_half, w - mask_size_half)
        cy = random.randint(mask_size_half, h - mask_size_half)

        x1 = cx - mask_size_half
        y1 = cy - mask_size_half
        x2 = cx + mask_size_half
        y2 = cy + mask_size_half

        img = T.to_tensor(img)
        img[:, y1:y2, x1:x2] = 0.0
        return T.to_pil_image(img)

# Final transform pipeline
train_transforms = transforms.Compose([
    transforms.Pad(4),  # Adds 4 pixels on each side (total +8 in height and width)
    transforms.RandomCrop(28),  # Random crop back to 28x28
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.40, hue=0.1),
    transforms.RandomRotation((-15., 15.), fill=0),
    #transforms.RandomHorizontalFlip(p=0.3),  # 30% chance to flip
    RandomCutout(mask_size=8, p=0.3),  # Randomly mask a square region
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean and std
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])


train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

batch_size = 100

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if cuda else {'batch_size': batch_size, 'shuffle': True}

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False) ## 28X28X16
        self.bn2 = nn.BatchNorm2d(16)

        self.dropout1 = nn.Dropout(0.05)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3,padding=1,bias=False) ## 28X28X16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3,padding=1,bias=False) ## 28X28X16
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout2 = nn.Dropout(0.05)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3,padding=0,bias=False) ##28X28X16
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 10, kernel_size=1,padding=0,bias=False) ##28X28X16
        self.bn6 = nn.BatchNorm2d(10)

        self.conv7 = nn.Conv2d(16, 10, kernel_size=1,padding=0,bias=False) ##28X28X16
        self.bn7 = nn.BatchNorm2d(10)

        self.conv2t = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False) ## 28X28X16
        self.bn2t = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(256, 10)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)

        #x = self.dropout2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)


        x = self.gap(x)

        x = x.view(-1, 10)
        #x = self.fc1(x)
        return x



# ---- Swell white pixels in a random square patch ----
def swell_white_pixels(img, box_size=7, dilation_kernel=3):
    c, h, w = img.shape
    y = torch.randint(0, h - box_size, (1,)).item()
    x = torch.randint(0, w - box_size, (1,)).item()

    patch = img[:, y:y+box_size, x:x+box_size].unsqueeze(0)
    dilated = F.max_pool2d(patch, kernel_size=dilation_kernel, stride=1, padding=dilation_kernel//2)
    dilated = dilated[:, :, :box_size, :box_size]

    img[:, y:y+box_size, x:x+box_size] = torch.max(
        img[:, y:y+box_size, x:x+box_size], dilated[0]
    )
    return img

# ---- Darken some white pixels inside the digit ----
def darken_white_pixels(img, darken_fraction=0.15):
    white_pixels = (img[0] > 0.5).nonzero(as_tuple=False)
    if len(white_pixels) == 0:
        return img
    n_to_darken = max(1, int(len(white_pixels) * darken_fraction))
    chosen_pixels = white_pixels[torch.randperm(len(white_pixels))[:n_to_darken]]
    for (y, x) in chosen_pixels:
        img[0, y, x] = img[0, y, x] * 0.2
    return img



def squeeze_digit_patch(img, patch_size=(6,6)):
    c, h, w = img.shape
    ph, pw = patch_size
    y = torch.randint(0, h - ph, (1,)).item()
    x = torch.randint(0, w - pw, (1,)).item()
    patch = img[:, y:y+ph, x:x+pw].unsqueeze(0)

    # Randomly choose direction per image
    direction = 'vertical' if torch.rand(1) < 0.5 else 'horizontal'

    if direction == 'vertical':
        squeezed = F.max_pool2d(patch, kernel_size=(ph,1), stride=1, padding=(0,0))
    else:
        squeezed = F.max_pool2d(patch, kernel_size=(1,pw), stride=1, padding=(0,0))

    squeezed = squeezed[:, :, :ph, :pw]
    img[:, y:y+ph, x:x+pw] = torch.max(img[:, y:y+ph, x:x+pw], squeezed[0])
    return img



def augment_batch_balanced(data, targets, num_per_label=10):
    num_classes = 10

    # --- Squeeze ---
    for label in range(num_classes):
        label_indices = (targets == label).nonzero(as_tuple=True)[0]
        if len(label_indices) >= num_per_label:
            chosen = label_indices[torch.randperm(len(label_indices))[:num_per_label]]
            for idx in chosen:
                data[idx] = squeeze_digit_patch(data[idx], patch_size=(6,6))

    # --- Swell ---
    for label in range(num_classes):
        label_indices = (targets == label).nonzero(as_tuple=True)[0]
        if len(label_indices) >= num_per_label:
            chosen = label_indices[torch.randperm(len(label_indices))[:num_per_label]]
            for idx in chosen:
                data[idx] = swell_white_pixels(data[idx])


    # --- Darken ---
    for label in range(num_classes):
        label_indices = (targets == label).nonzero(as_tuple=True)[0]
        if len(label_indices) >= num_per_label:
            chosen = label_indices[torch.randperm(len(label_indices))[:num_per_label]]
            for idx in chosen:
                data[idx] = darken_white_pixels(data[idx])



    return data



def show_misclassified(model, test_loader, device, classes, max_images=100):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            mismatches = preds != labels
            for img, label, pred in zip(images[mismatches], labels[mismatches], preds[mismatches]):
                misclassified_images.append(img.cpu())
                misclassified_labels.append(label.item())
                misclassified_preds.append(pred.item())
                if len(misclassified_images) >= max_images:
                    break
            if len(misclassified_images) >= max_images:
                break

    # ---- Plot them ----
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle("Misclassified Images", fontsize=20)
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_images):
            img = misclassified_images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"T:{classes[misclassified_labels[i]]}\nP:{classes[misclassified_preds[i]]}",
                         fontsize=8, color="red")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    #data = augment_batch_balanced(data, target,5)  # <--- our new augmentation
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    print(torch.__version__)

    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure(figsize=(10, 8))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(f"Label: {batch_label[i].item()}")
        plt.xticks([])
        plt.yticks([])

    plt.show()

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Net3().to(device)
    summary(model, input_size=(1, 28, 28))

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # large learning rate
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, epochs=20, steps_per_epoch=len(train_loader))

    #optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam usually needs smaller LR
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)



    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # Optimizer and Scheduler - OneCycleLR is usually very effective
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5) # Added weight decay
    scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20, steps_per_epoch=len(train_loader),
                           pct_start=0.3, div_factor=10, final_div_factor=100) # Increased epochs and adjusted OneCycleLR
    # New Line
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20

    print("\nStarting Training...")
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.suptitle("Training and Test Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    classes = [str(i) for i in range(10)]  # MNIST labels 0-9
    show_misclassified(model, test_loader, device, classes, max_images=100)
