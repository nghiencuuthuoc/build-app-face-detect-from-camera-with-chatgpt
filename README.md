<table align="center">
  <td>
    <a href="https://colab.research.google.com/github/nghiencuuthuoc/PharmApp/blob/master/notebook/PharmApp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  </td>
  <td>
    <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/nghiencuuthuoc/PharmApp/blob/master/notebook/PharmApp.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
  </td>
</table>

![](./images/PharmApp-logo.png)
# **PharmApp** - 🧠 AI for Research and Development Pharmaceuticals
Copyright 2025 | Nghiên Cứu Thuốc | RnD_Pharma_Plus

Email: nghiencuuthuoc@gmail.com | Web: nghiencuuthuoc.com | FB: facebook.com/nghiencuuthuoc 

LinkedIn: linkedin.com/in/nghiencuuthuoc | Zalo: +84888999311 | WhatsAapp: +84888999311

Twitter: x.com/nghiencuuthuoc | YT: youtube.com/@nghiencuuthuoc 


## ⚡️Ứng dụng AI ChatGPT Xây Dựng App Trích xuất khuôn mặt và làm trắng da từ Camera

**Mục Tiêu:** 
Hướng dẫn thực hành giúp sinh viên hiểu và xây dựng ứng dụng nhận diện khuôn mặt và làm sáng da, mịn màng qua camera thời gian thực. 

**Công Cụ Cần Chuẩn Bị:**
1. **Python**: Ngôn ngữ lập trình sử dụng để xây dựng ứng dụng.
2. **Thư Viện**:
   - `opencv-python`: Dùng để xử lý hình ảnh và nhận diện khuôn mặt.
   - `mediapipe`: Thư viện hỗ trợ nhận diện khuôn mặt và các bộ phận cơ thể.
   - `Pillow (PIL)`: Dùng để xử lý và chỉnh sửa hình ảnh (như làm sáng da).
3. **Camera**: Để thu hình ảnh và nhận diện khuôn mặt trực tiếp.

### Các Bước Xây Dựng Code

**1. Cài đặt các thư viện cần thiết**

Trước hết, sinh viên cần cài đặt các thư viện sau qua pip:

```bash
pip install opencv-python mediapipe pillow
```

**2. Sử Dụng Mediapipe để Nhận Diện Khuôn Mặt**

- Mediapipe là một thư viện mạnh mẽ của Google giúp nhận diện khuôn mặt, cử chỉ tay, và nhiều đối tượng khác trong thời gian thực.
- Chúng ta sẽ dùng Mediapipe để nhận diện khuôn mặt và khu vực từ vai đến đầu.

```python
import cv2
import numpy as np
import mediapipe as mp

# Khởi tạo mô hình nhận diện khuôn mặt
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển ảnh sang định dạng RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Mở rộng khung bao để bao phủ cả vai và đầu
                extended_y = max(0, y - int(h / 2))
                extended_h = h + int(h / 2)

                # Cắt ảnh vuông từ vai đến đầu
                square_size = max(w, extended_h)
                extended_x = max(0, x - (square_size - w) // 2)
                body_img = frame[extended_y:extended_y+square_size, extended_x:extended_x+square_size]

                # Hiển thị ảnh
                cv2.imshow('Body Detection', body_img)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
```

**Giải Thích:**
- Đoạn mã trên sử dụng Mediapipe để nhận diện khuôn mặt và cắt phần khuôn mặt từ vai lên, tạo một vùng ảnh vuông bao quanh khuôn mặt.
- Bước tiếp theo là làm sáng và làm mịn da trong phần này.

**3. Làm Sáng Da và Làm Mịn Khuôn Mặt**

Để có làn da mịn màng và sáng hơn, chúng ta sẽ sử dụng phương pháp **tăng độ sáng** và **tăng độ tương phản** của ảnh. Cụ thể, sử dụng thư viện Pillow để điều chỉnh ảnh.

```python
from PIL import Image, ImageEnhance

# Chuyển ảnh từ OpenCV sang PIL để dễ dàng chỉnh sửa
pil_img = Image.fromarray(cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB))

# Tăng độ sáng cho ảnh
enhancer = ImageEnhance.Brightness(pil_img)
pil_img = enhancer.enhance(1.4)  # Tăng độ sáng lên 1.4 lần

# Tăng độ tương phản để làm rõ các chi tiết
contrast_enhancer = ImageEnhance.Contrast(pil_img)
pil_img = contrast_enhancer.enhance(1.2)  # Tăng độ tương phản lên 1.2 lần

# Hiển thị kết quả
pil_img.show()
```

**Giải Thích:**
- **ImageEnhance.Brightness**: Dùng để làm sáng ảnh, giúp làn da trở nên sáng và đều màu hơn.
- **ImageEnhance.Contrast**: Dùng để tăng độ tương phản, làm ảnh trông sắc nét hơn và cải thiện chi tiết.

**4. Cắt Khuôn Mặt thành Hình Tròn**

Cuối cùng, chúng ta sẽ áp dụng một lớp mặt nạ hình tròn lên ảnh đã xử lý để tạo ra một ảnh với khuôn mặt được cắt gọn trong một vòng tròn.

```python
# Tạo mặt nạ hình tròn
mask = Image.new('L', pil_img.size, 0)
draw = ImageDraw.Draw(mask)

# Tạo vòng tròn để cắt ảnh
circle_radius = int(pil_img.size[0] / 2)
center_x, center_y = pil_img.size[0] // 2, pil_img.size[1] // 2

draw.ellipse((center_x - circle_radius, center_y - circle_radius,
              center_x + circle_radius, center_y + circle_radius), fill=255)

# Áp dụng mặt nạ hình tròn
pil_img.putalpha(mask)

# Tạo ảnh có nền trong suốt và dán ảnh vào đó
circle_frame = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
circle_frame.paste(pil_img, (0, 0), mask=pil_img)

# Hiển thị ảnh cuối cùng
circle_frame.show()
```

**Giải Thích:**
- Chúng ta sử dụng **PIL ImageDraw** để vẽ một vòng tròn và dùng mặt nạ này để tạo ảnh khuôn mặt cắt hình tròn.

### Tổng Kết

1. **Nhận diện khuôn mặt**: Dùng Mediapipe để nhận diện khuôn mặt và cắt vùng từ vai lên đầu.
2. **Làm sáng và mịn da**: Tăng độ sáng và độ tương phản của ảnh để làn da sáng mịn, đẹp hơn.
3. **Cắt khuôn mặt thành hình tròn**: Dùng Pillow để tạo một mặt nạ hình tròn cho khuôn mặt và áp dụng lên ảnh.

### Lời Khuyên cho Sinh Viên

- Hãy thực hành từng bước để hiểu cách xử lý ảnh và sử dụng các thư viện Python.
- Đảm bảo làm quen với các hàm trong `mediapipe`, `opencv` và `PIL` để có thể tối ưu hóa quá trình xây dựng ứng dụng nhận diện khuôn mặt và chỉnh sửa ảnh.
- Cải tiến thêm các hiệu ứng như làm mịn da hay tạo thêm các hiệu ứng sáng tạo cho ảnh.

---

Hy vọng bài viết này sẽ giúp sinh viên hiểu rõ hơn về quá trình xây dựng ứng dụng xử lý ảnh và nhận diện khuôn mặt, cũng như cải thiện kỹ năng lập trình với các thư viện Python!
