import cv2
import mediapipe as mp
import math  # 거리 계산을 위해 math 모듈 추가

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

    # ---- 이 부분을 새로운 거리 계산 로직으로 교체하세요 ----
    def is_fist(self, hand_landmarks):
        """손의 방향(회전)과 무관하게 손목과의 거리로 주먹 여부를 판단합니다."""
        tips = [8, 12, 16, 20] # 검지, 중지, 약지, 새끼 끝 마디
        pips = [6, 10, 14, 18] # 두 번째 마디
        wrist = hand_landmarks.landmark[0] # 손목 랜드마크 (0번 기준점)
        
        folded_count = 0
        for tip, pip in zip(tips, pips):
            tip_lm = hand_landmarks.landmark[tip]
            pip_lm = hand_landmarks.landmark[pip]
            
            # math.hypot(피타고라스 정리)를 이용해 손목부터 각 마디까지의 직선 거리 계산
            dist_tip = math.hypot(tip_lm.x - wrist.x, tip_lm.y - wrist.y)
            dist_pip = math.hypot(pip_lm.x - wrist.x, pip_lm.y - wrist.y)
            
            # 손가락 끝(Tip)이 두 번째 마디(PIP)보다 손목에 더 가까워지면 접힌 것으로 판단
            if dist_tip < dist_pip:
                folded_count += 1
                
        # 4개 중 3개 이상 접혔을 때 True 반환
        return folded_count >= 3
    # -----------------------------------------------------

    def get_hand_info(self, frame, results):
        hands_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label 
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                
                hands_list.append({
                    "label": label,
                    "center": (cx, cy),
                    "landmarks": hand_landmarks,
                    "is_fist": self.is_fist(hand_landmarks)
                })
        return hands_list