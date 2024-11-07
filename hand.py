import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# initialize components
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cam = cv2.VideoCapture(0)

# setup hand gesture recognizer
with open("resources/gesture_recognizer.task", 'rb') as file:
    model_data = file.read()
recognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_buffer = model_data)))

while True:
    success, img = cam.read()  # capture frame-by-frame
    if not success:
        break

    img = cv2.flip(img, 1)  # flip the image for natural mirroring
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:  # draw landmarks if hands are detected
            mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        recognition_result = recognizer.recognize(mp_image)
        if recognition_result.gestures and recognition_result.gestures[0]:
            gesture_text = f"{recognition_result.gestures[0][0].category_name}"
            cv2.putText(img, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition", img)

    if cv2.waitKey(1) != -1:  # exit on any key press
        break

# cleanup
cam.release()
cv2.destroyAllWindows()
