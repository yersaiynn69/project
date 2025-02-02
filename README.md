# project
# В кратце сохраняешь веса после обучения: 
torch.save(model.state_dict(), "face_recognition_model.pth")
print("Модель сохранена.")

# загружаешь обученную модель
model.load_state_dict(torch.load("face_recognition_model.pth", map_location=device))
model.eval()  #обьязательно  включаете eval-режим

дальше настраиваем real time инференс веб камеры
добавляем обнаружение лица через cv2.CascadeClassifier + предсказание.
import cv2

# подключаем OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# функция для real-time предсказания
def recognize_faces(model, device, threshold=0.5):
    cap = cv2.VideoCapture(0)  # веб-камера (или можно указать путь к видео)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # преобразуем в ЧБ
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # обнаружение лиц
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # вырезаем лицо
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face = transform(face).unsqueeze(0).to(device)  # преобразуем
            
            with torch.no_grad():
                embedding = model(face)  # получаем эмбеддинг лица
            
            # пока просто рисуем рамку (по факту тут должен быть поиск в базе)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # нажми "q" для выхода
            break
    
    cap.release()
    cv2.destroyAllWindows()

# запуск real-time распознавания
recognize_faces(model, device)

Оптимизация:
Надо перевести в ONNX – это даст ускорение для CPU
Будем использовать TensorRT/OpenVino – для максимального ускорения
Запустить на FP16 ( уменьшает размер модели не теряя точность

Экспорт в ONNX: 
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
torch.onnx.export(model, dummy_input, "face_model.onnx", opset_version=11)
print("Модель экспортирована в ONNX.")
после этого можно запустить через OpenVINO / TensorRT.


