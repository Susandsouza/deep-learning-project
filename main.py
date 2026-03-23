import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs/generated_images", exist_ok=True)

# =========================================
# 🔥 PART A: CNN (CIFAR-10)
# =========================================
print("\n===== CNN TRAINING =====")

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

cnn_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

for epoch in range(2):  # keep small for fast run
    total_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"CNN Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================================
# 🔁 PART B: RNN / LSTM / GRU
# =========================================
print("\n===== RNN / LSTM / GRU =====")

vocab_size = 5000
seq_len = 50

X = torch.randint(0, vocab_size, (500, seq_len)).to(device)
y = torch.randint(0, 2, (500,)).float().to(device)

class RNNModel(nn.Module):
    def __init__(self, rnn_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(64, 32, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(64, 32, batch_first=True)
        else:
            self.rnn = nn.RNN(64, 32, batch_first=True)

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

def train_rnn(rnn_type):
    model = RNNModel(rnn_type).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print(f"{rnn_type} Final Loss: {loss.item():.4f}")

train_rnn("RNN")
train_rnn("LSTM")
train_rnn("GRU")

# =========================================
# 🎨 PART C: GAN (Fashion-MNIST)
# =========================================
print("\n===== GAN TRAINING =====")

transform = transforms.ToTensor()
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

from torchvision.utils import save_image

for epoch in range(3):  # small epochs for fast run
    for real, _ in loader:
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, 100).to(device)
        fake = G(noise)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        loss_D = criterion(D(real), real_labels) + \
                 criterion(D(fake.detach()), fake_labels)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        loss_G = criterion(D(fake), real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"GAN Epoch {epoch+1} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # Save generated images
    save_image(fake.view(-1,1,28,28), f"outputs/generated_images/epoch_{epoch}.png")

print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY")