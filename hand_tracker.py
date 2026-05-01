import cv2
import mediapipe as mp

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
        """손을 찾고 랜드마크를 반환합니다."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def is_fist(self, hand_landmarks):
        """4개 손가락이 접혔는지 확인하여 주먹 여부를 판단합니다."""
        tips = [8, 12, 16, 20] # 검지, 중지, 약지, 새끼 끝
        pips = [6, 10, 14, 18] # 두 번째 마디
        
        folded_count = 0
        for tip, pip in zip(tips, pips):
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                folded_count += 1
        return folded_count >= 3

    def get_hand_info(self, frame, results):
        """손의 위치(좌표)와 어느 쪽 손인지 정보를 추출합니다."""
        hands_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label # 'Left' or 'Right'
                h, w, _ = frame.shape
                # 손목(0번) 좌표
                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                
                hands_list.append({
                    "label": label,
                    "center": (cx, cy),
                    "landmarks": hand_landmarks,
                    "is_fist": self.is_fist(hand_landmarks)
                })
        return hands_list