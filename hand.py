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


# some constants
BGR_SKIN_LT = get_bgr_color("peachpuff")
BGR_SKIN_DK = get_bgr_color("burlywood")
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500
IMAGE_DIMENSIONS = (IMAGE_WIDTH, IMAGE_HEIGHT)


def run():
    # initialize components
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cam = cv2.VideoCapture(0)  # change number to cycle between multiple input feeds
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)

    while True:
        success, img = cam.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)  # flip for natural mirroring

        for hl in hands.process(img_rgb).multi_hand_landmarks or []:
            canvas = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255
            
            # draw connections with realistic thickness and shading
            avg_z = sum(lm.z for lm in hl.landmark) / len(hl.landmark)
            depth_factor = max(5, min(15, 15 + (avg_z + 0.2) * (5 - 15) / (-0.005 + 0.2)))
            #print(f"avg_z: {avg_z} - depth_factor: {depth_factor}")

            for lm in hl.landmark:
                x, y = int(lm.x * IMAGE_WIDTH), int(lm.y * IMAGE_HEIGHT)
                #depth_shade = int((1 - lm.z) * 50)  # depth-based shading effect
                cv2.circle(canvas, (x, y), int(depth_factor * 2), BGR_SKIN_LT, -1)
            
            mp_draw.draw_landmarks(
                canvas, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(
                    color=BGR_SKIN_DK, thickness=int(depth_factor * 2), circle_radius=0),
                mp_draw.DrawingSpec(
                    color=BGR_SKIN_LT, thickness=int(depth_factor * 3), circle_radius=0))

            cv2.imshow("Hand on White Background", canvas)

        cv2.imshow("Original Webcam Feed", img)  # display original feed

        if cv2.waitKey(1) != -1:  # exit on any key press
            break

    # cleanup
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
