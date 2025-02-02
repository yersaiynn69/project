### 1. train.py (Обучение модели)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import os
from dataset import FaceDataset
from model import FaceEmbeddingModel, ArcFaceLoss

# Гиперпараметры
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
DATASET_PATH = "./dataset"

# Датасет и DataLoader
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = FaceDataset(DATASET_PATH, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Модель и оптимизатор
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmbeddingModel().to(device)
criterion = ArcFaceLoss(num_classes=len(dataset.label_dict))
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Обучение
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Сохранение модели
torch.save(model.state_dict(), "face_model.pth")
print("Модель сохранена!")


### 2. create_embeddings.py (Создание базы лиц)
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from model import FaceEmbeddingModel

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmbeddingModel().to(device)
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

# Трансформации
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Создание базы эмбеддингов
face_database = {}
DATASET_PATH = "./dataset"

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    if os.path.isdir(person_dir):
        images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
        embeddings = []
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image).cpu().numpy()
            embeddings.append(embedding)
        face_database[person] = np.mean(embeddings, axis=0)

# Сохранение базы
np.save("face_database.npy", face_database)
print("База лиц сохранена!")


### 3. real_time_recognition.py (Распознавание в реальном времени)
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from model import FaceEmbeddingModel

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmbeddingModel().to(device)
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

# Загрузка базы эмбеддингов
face_database = np.load("face_database.npy", allow_pickle=True).item()

# Трансформации
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Поиск лица в базе
def recognize_face(embedding):
    best_match = "Unknown"
    best_score = 0.0
    for name, db_embedding in face_database.items():
        similarity = cosine_similarity(embedding.cpu().numpy(), db_embedding)[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = name
    return best_match if best_score > 0.5 else "Unknown"

# Видеопоток
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face = transform(face).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(face)
        
        name = recognize_face(embedding)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
