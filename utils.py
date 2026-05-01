import cv2
import numpy as np
import os

def load_overlay_img(filename, size=(150, 150)):
    """이미지를 불러와 크기를 조절합니다."""
    if os.path.exists(filename):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED) # 투명도 포함 불러오기
        return cv2.resize(img, size)
    else:
        # 파일이 없을 경우 임시로 빈 사각형 생성
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(img, "No Image", (10, size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

def overlay_transparent(background, overlay, x, y):
    """배경 이미지 위에 덮어씌울 이미지를 합성합니다 (투명도 지원)."""
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    # 화면 경계 처리
    x1, x2 = max(x, 0), min(x + w, bg_w)
    y1, y2 = max(y, 0), min(y + h, bg_h)
    
    ol_x1, ol_x2 = x1 - x, x2 - x
    ol_y1, ol_y2 = y1 - y, y2 - y

    overlay_crop = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    
    # 4채널(Alpha) 이미지인 경우 투명도 처리
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha * overlay_crop[:, :, c] +
                                          (1.0 - alpha) * background[y1:y2, x1:x2, c])
    else:
        background[y1:y2, x1:x2] = overlay_crop[:, :, :3]
        
    return background