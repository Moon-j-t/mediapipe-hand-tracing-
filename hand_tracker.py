import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, max_hands=2, detection_con=0.7, track_con=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def is_fist(self, hand_landmarks):
        """손목과의 거리를 계산하여 주먹 여부를 판단합니다."""
        tips = [8, 12, 16, 20] 
        pips = [6, 10, 14, 18] 
        wrist = hand_landmarks.landmark[0] 
        
        folded_count = 0
        for tip, pip in zip(tips, pips):
            tip_lm = hand_landmarks.landmark[tip]
            pip_lm = hand_landmarks.landmark[pip]
            dist_tip = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
            dist_pip = math.hypot(pip_lm.x - wrist.x, pip_lm.y - wrist.y)
            if dist_tip < dist_pip:
                folded_count += 1
        return folded_count >= 3

    def is_walking_fingers(self, hand_landmarks):
        """검지와 중지가 펴져 있고, 손목보다 아래를 향하며, 나머지는 접혀있는 형태 판별"""
        wrist = hand_landmarks.landmark[0]
        
        # 1. 방향 조건: 검지(8)와 중지(12) 끝이 손목보다 화면 아래(y값이 더 큼)에 있는가?
        if hand_landmarks.landmark[8].y < wrist.y or hand_landmarks.landmark[12].y < wrist.y:
            return False 

        # 2. 펴짐/접힘 조건 확인
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        is_extended = []
        
        for tip, pip in zip(tips, pips):
            tip_lm = hand_landmarks.landmark[tip]
            pip_lm = hand_landmarks.landmark[pip]
            dist_tip = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
            dist_pip = math.hypot(pip_lm.x - wrist.x, pip_lm.y - wrist.y)
            is_extended.append(dist_tip > dist_pip)
            
        # 3. 검지(0), 중지(1)는 펴져있고(True) / 약지(2), 새끼(3)는 접혀있는가(False)?
        index_middle_extended = is_extended[0] and is_extended[1]
        ring_pinky_folded = not is_extended[2] and not is_extended[3]
        
        return index_middle_extended and ring_pinky_folded

    def get_hand_info(self, frame, results):
        hands_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # --- 추가된 부분: 손의 구조(뼈대와 연결선)를 화면에 그립니다 ---
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # -----------------------------------------------------------
                
                label = handedness.classification[0].label 
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                
                hands_list.append({
                    "label": label,
                    "center": (cx, cy),
                    "landmarks": hand_landmarks,
                    "is_fist": self.is_fist(hand_landmarks),
                    "is_walking": self.is_walking_fingers(hand_landmarks) 
                })
        return hands_list