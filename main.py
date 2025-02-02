import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics.pairwise import cosine_similarity

# Гиперпараметры
BATCH_SIZE = 16  # Меньший батч уменьшает нагрузку на память
EPOCHS = 25
LEARNING_RATE = 0.001
IMG_SIZE = 112
EMBEDDING_SIZE = 256
GRAD_ACCUMULATION_STEPS = 2  # Аккумулируем градиенты для имитации большего batch size

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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_dict = {}
        self._load_images()

    def _load_images(self):
        label_id = 0
        for person in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person)
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

# ArcFace Loss (замена Triplet Loss)
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE, num_classes=5000, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings, labels):
        cosine = nn.functional.linear(nn.functional.normalize(embeddings), nn.functional.normalize(self.weights))
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        theta = theta + self.margin
        logits = torch.cos(theta) * self.scale
        return nn.CrossEntropyLoss()(logits, labels)

# Модель (EfficientNet-b0)
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE):
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
        return nn.functional.normalize(x, p=2, dim=1)  # Нормализуем вектор

# Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmbeddingModel().to(device)
criterion = ArcFaceLoss(num_classes=5000)  # Нужно задать реальное число классов в датасете
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()  # Для FP16 обучения
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Функция обучения
def train(model, dataloader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            with autocast():  # FP16 для снижения нагрузки
                embeddings = model(images)
                loss = criterion(embeddings, labels) / GRAD_ACCUMULATION_STEPS  # Делим на шаги аккумуляции
            
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:  
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        scheduler.step(total_loss)  # Автоматическое уменьшение LR

# Подготовка данных
DATASET_PATH = "./lfw"
if os.path.exists(DATASET_PATH):
    dataset = FaceDataset(DATASET_PATH, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

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
# Пути к вашим датасетам
dataset_paths = ["./dataset1", "./dataset2", "./dataset3"]

# Создание объединённого датасета
combined_dataset = FaceDataset(dataset_paths, transform)

# DataLoader
dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    train(model, dataloader, criterion, optimizer)
else:
    print("Ошибка: Указанный путь к датасету не существует.")
