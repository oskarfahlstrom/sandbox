import cv2
import mediapipe as mp

# initialize mediapipe hands and cv2
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()  # capture frame-by-frame
    if not success:
        break

    img = cv2.flip(img, 1)  # flip the image for natural mirroring
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for hl in results.multi_hand_landmarks or []:  # draw landmarks if hands are detected
        mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Recognition", img)

    if cv2.waitKey(1) != -1:  # exit on any key press
        break

# cleanup
cam.release()
cv2.destroyAllWindows()
