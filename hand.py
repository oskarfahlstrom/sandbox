import cv2
import numpy as np
import mediapipe as mp
from matplotlib import colors


def get_bgr_color(name):
    """Convert a named color to BGR format."""
    return tuple(int(c * 255) for c in reversed(colors.to_rgb(name)))

def adjust_bgr_color(color, depth_shade):
    """Adjust each channel and clamp between 0 and 255."""
    return tuple(max(0, min(255, c - depth_shade)) for c in color)


# initialize components
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cam = cv2.VideoCapture(0)

# some constants
BGR_SKIN = get_bgr_color("peachpuff")
BGR_SHADOW = get_bgr_color("saddlebrown")
IMAGE_WIDTH = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
IMAGE_HEIGHT = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
IMAGE_DIMENSIONS = (IMAGE_WIDTH, IMAGE_HEIGHT)


while True:
    success, img = cam.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)  # flip for natural mirroring

    for hl in hands.process(img_rgb).multi_hand_landmarks or []:
        canvas = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255
        
        for lm in hl.landmark:
            x, y = int(lm.x * IMAGE_WIDTH), int(lm.y * IMAGE_HEIGHT)
            depth_shade = int((1 - lm.z) * 50)  # depth-based shading effect
            cv2.circle(canvas, (x, y), 8, adjust_bgr_color(BGR_SKIN, depth_shade), -1)
        
        # draw connections with realistic thickness and shading
        mp_draw.draw_landmarks(
            canvas, hl, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=BGR_SHADOW, thickness=9, circle_radius=0),
            mp_draw.DrawingSpec(color=BGR_SKIN, thickness=6, circle_radius=0))

        cv2.imshow("Hand on White Background", canvas)

    #cv2.imshow("Original Webcam Feed", img)  # display original feed

    if cv2.waitKey(1) != -1:  # exit on any key press
        break

# cleanup
cam.release()
cv2.destroyAllWindows()
