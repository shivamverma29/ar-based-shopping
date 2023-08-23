import os
import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as mp


def VideoCamera():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = PoseDetector()

    shirtFolderPath = "Shirts"
    listShirts = os.listdir(shirtFolderPath)

    fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
    shirtRatioHeightWidth = 581 / 440
    imageNumber = 0

    thumbs_up = False
    prev_thumb_state = False

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lmList = []
                for lm in hand_landmarks.landmark:
                    lmList.append([lm.x, lm.y, lm.z])  # Append x, y, z coordinates

                # Here, you can use the lmList to access the hand landmarks
                # For example, to get the y-coordinate of the thumb tip:
                thumb_y = lmList[4][1]
                middle_y = lmList[12][1]
                thumb_z = lmList[4][2]
                wrist_z = lmList[0][2]

                if thumb_y < middle_y and thumb_z < wrist_z:
                    thumbs_up = True
                else:
                    thumbs_up = False

        if thumbs_up and not prev_thumb_state:
            # Switch to the next t-shirt
            imageNumber = (imageNumber + 1) % len(listShirts)
            prev_thumb_state = True
        elif not thumbs_up and prev_thumb_state:
            prev_thumb_state = False

        img = detector.findPose(img)

        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
            imgShirt = cv2.imread(
                os.path.join(shirtFolderPath, listShirts[imageNumber]),
                cv2.IMREAD_UNCHANGED,
            )

            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            imgShirt = cv2.resize(
                imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth))
            )
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)

            try:
                # Overlay shirt on the webcam feed
                for c in range(0, 3):
                    img[
                        lm12[1] - offset[1] : lm12[1] - offset[1] + imgShirt.shape[0],
                        lm12[0] - offset[0] : lm12[0] - offset[0] + imgShirt.shape[1],
                        c,
                    ] = imgShirt[:, :, c] * (imgShirt[:, :, 3] / 255.0) + img[
                        lm12[1] - offset[1] : lm12[1] - offset[1] + imgShirt.shape[0],
                        lm12[0] - offset[0] : lm12[0] - offset[0] + imgShirt.shape[1],
                        c,
                    ] * (
                        1.0 - imgShirt[:, :, 3] / 255.0
                    )
            except:
                pass

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
