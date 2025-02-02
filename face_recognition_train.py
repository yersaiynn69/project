import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
import os
from torch.cuda.amp import autocast, GradScaler

# Гиперпараметры
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
IMG_SIZE = 112
EMBEDDING_SIZE = 256
GRAD_ACCUMULATION_STEPS = 2

# Аугментации
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Датасет
class FaceDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_dict = {}
        self._load_images(root_dirs)

    def _load_images(self, root_dirs):
        label_id = 0
        for root_dir in root_dirs:
            for person in os.listdir(root_dir):
                person_dir = os.path.join(root_dir, person)
                if os.path.isdir(person_dir):
                    if person not in self.label_dict:
                        self.label_dict[person] = label_id
                        label_id += 1
                    for img_name in os.listdir(person_dir):
                        img_path = os.path.join(person_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.label_dict[person])

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

# ArcFace Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings, labels):
        cosine = nn.functional.linear(nn.functional.normalize(embeddings), nn.functional.normalize(self.weights))
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        theta += self.margin
        logits = torch.cos(theta) * self.scale
        return nn.CrossEntropyLoss()(logits, labels)

# Модель
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(FaceEmbeddingModel, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.num_features),
            nn.Linear(self.model.num_features, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

# Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_paths = ["./dataset1", "./dataset2", "./dataset3"]
dataset = FaceDataset(dataset_paths, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True)

num_classes = len(dataset.label_dict)
model = FaceEmbeddingModel(embedding_size=EMBEDDING_SIZE).to(device)
criterion = ArcFaceLoss(embedding_size=EMBEDDING_SIZE, num_classes=num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Функция обучения
def train(model, dataloader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                embeddings = model(images)
                loss = criterion(embeddings, labels) / GRAD_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0 or step == len(dataloader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

# Запуск обучения
train(model, dataloader, criterion, optimizer)
