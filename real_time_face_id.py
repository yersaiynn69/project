import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация модели
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка каскада для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Трансформации для модели
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# База данных лиц (пример)
face_database = {
    "Alice": torch.rand(1, EMBEDDING_SIZE).to(device),  # Заглушки, замените на реальные эмбеддинги
    "Bob": torch.rand(1, EMBEDDING_SIZE).to(device)
}

def recognize_face(embedding):
    best_match = None
    best_score = 0.0
    
    for name, db_embedding in face_database.items():
        similarity = cosine_similarity(embedding.cpu().numpy(), db_embedding.cpu().numpy())[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = name
    
    return best_match if best_score > 0.5 else "Unknown"  # Порог можно настроить

# Запуск видеопотока
cap = cv2.VideoCapture(0)
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
        
        name = recognize_face(embedding)  # Поиск в базе
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
