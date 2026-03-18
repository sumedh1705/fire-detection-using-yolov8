import os
import smtplib
from pathlib import Path

import cv2
import playsound
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "best.pt"
ALARM_PATH = BASE_DIR / "assets" / "alarm.mp3"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.6


def play_alarm() -> None:
    if not ALARM_PATH.exists():
        print(f"Warning: alarm file not found at {ALARM_PATH}")
        return

    try:
        playsound.playsound(str(ALARM_PATH), block=False)
    except Exception as exc:
        print(f"Warning: could not play alarm sound: {exc}")


def send_sos_email() -> None:
    sender_email = os.getenv("FIRE_ALERT_SENDER_EMAIL")
    receiver_email = os.getenv("FIRE_ALERT_RECEIVER_EMAIL")
    password = os.getenv("FIRE_ALERT_APP_PASSWORD")

    if not sender_email or not receiver_email or not password:
        print("Email alert skipped: set FIRE_ALERT_SENDER_EMAIL, FIRE_ALERT_RECEIVER_EMAIL, and FIRE_ALERT_APP_PASSWORD.")
        return

    message = (
        "Subject: Fire Detection Alert\n\n"
        "SOS! Fire detected by the Real-Time Fire Detection System."
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        print("SOS email sent successfully.")
    except Exception as exc:
        print(f"Error sending SOS email: {exc}")


def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Place your trained model at weights/best.pt."
        )

    return YOLO(str(MODEL_PATH))


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Unable to open the webcam. Check the camera connection and permissions.")

    return cap


def detect_fire(result) -> bool:
    return result.boxes is not None and len(result.boxes) > 0


def main() -> None:
    model = load_model()
    cap = open_camera()
    alarm_triggered = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: unable to capture frame.")
                break

            results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, show=False, verbose=False)
            result = results[0]
            annotated_frame = result.plot()

            if detect_fire(result):
                cv2.putText(
                    annotated_frame,
                    "FIRE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                print("Fire detected!")
                if not alarm_triggered:
                    play_alarm()
                    send_sos_email()
                    alarm_triggered = True
            else:
                alarm_triggered = False

            cv2.imshow("Fire Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
