import cv2
import numpy as np
import mediapipe as mp

from constants import (BGR_SKIN_DK, BGR_SKIN_LT, IMAGE_DIMENSIONS, IMAGE_HEIGHT, IMAGE_WIDTH)


def run(cam_feed: int = 0):
    """Launch a hand recognition session. Pass webcam feed id or use 0 as default."""
    # initialize components
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cam = cv2.VideoCapture(cam_feed)  # change number to cycle between multiple input feeds
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)

    while True:
        success, img = cam.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)  # flip for natural mirroring

        for hl in hands.process(img_rgb).multi_hand_landmarks or []:
            canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)  # black background
            
            # draw connections with realistic thickness and shading
            avg_z = sum(lm.z for lm in hl.landmark) / len(hl.landmark)
            depth_factor = max(5, min(15, 15 + (avg_z + 0.2) * (5 - 15) / (-0.005 + 0.2)))

            for lm in hl.landmark:
                x, y = int(lm.x * IMAGE_WIDTH), int(lm.y * IMAGE_HEIGHT)
                cv2.circle(canvas, (x, y), int(depth_factor * 2), BGR_SKIN_LT, -1)
            
            mp_draw.draw_landmarks(
                canvas, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(
                    color=BGR_SKIN_DK, thickness=int(depth_factor * 2), circle_radius=0),
                mp_draw.DrawingSpec(
                    color=BGR_SKIN_LT, thickness=int(depth_factor * 3), circle_radius=0))

            rotated_canvas = rotate_image(image=canvas, angle=45)
            cv2.imshow("Hand on Black Background", rotated_canvas)

        if cv2.waitKey(1) != -1:  # exit on any key press
            break

    # cleanup
    cam.release()
    cv2.destroyAllWindows()


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate the canvas content while keeping the frame size unchanged."""
    # create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(
        (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2), angle, 1.0)

    # perform the rotation without changing the frame size
    params = {
        "flags": cv2.INTER_LINEAR, 
        "borderMode":cv2.BORDER_CONSTANT, 
        "borderValue":(0, 0, 0)}
    return cv2.warpAffine(image, rotation_matrix, IMAGE_DIMENSIONS, **params)


if __name__ == "__main__":
    run(0)  # webcam feed 0