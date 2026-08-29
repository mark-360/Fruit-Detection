from ultralytics import YOLO
import cv2

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = r"D:\Fruit Detection\runs\baseline_model\weights\best.pt"

model = YOLO(MODEL_PATH)

# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0

CONFIDENCE = 0.4
IMAGE_SIZE = 640

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

WINDOW_NAME = "Fruit Detection"

# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

# Request high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Optional: request better FPS if supported
cap.set(cv2.CAP_PROP_FPS, 60)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 60)
print("YOLO11 FRUIT DETECTION")
print("=" * 60)

print(f"Camera resolution: {actual_width} x {actual_height}")
print(f"Camera FPS: {actual_fps:.1f}")
print()

print("Q = Quit")
print("F = Toggle Fullscreen")
print()

# ============================================================
# FULLSCREEN WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

fullscreen = True

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    results = model(
        frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    # ========================================================
    # DRAW YOLO BOXES
    # ========================================================

    annotated_frame = result.plot()

    # ========================================================
    # COUNT DETECTED FRUITS
    # ========================================================

    counts = {}

    if result.boxes is not None:

        class_ids = result.boxes.cls.tolist()

        for class_id in class_ids:

            class_id = int(class_id)

            fruit_name = model.names[class_id]

            if fruit_name not in counts:
                counts[fruit_name] = 0

            counts[fruit_name] += 1

    # ========================================================
    # DISPLAY ONLY DETECTED FRUITS
    # ========================================================

    if counts:

        # ----------------------------------------------------
        # Create dark background panel
        # ----------------------------------------------------

        panel_height = 60 + (45 * len(counts))

        overlay = annotated_frame.copy()

        cv2.rectangle(
            overlay,
            (20, 20),
            (420, panel_height),
            (0, 0, 0),
            -1
        )

        # Transparency
        alpha = 0.55

        annotated_frame = cv2.addWeighted(
            overlay,
            alpha,
            annotated_frame,
            1 - alpha,
            0
        )

        # ----------------------------------------------------
        # FRUIT COUNTS
        # ----------------------------------------------------

        y_position = 60

        for fruit_name, count in counts.items():

            text = f"{fruit_name.upper()} : {count}"

            cv2.putText(
                annotated_frame,
                text,
                (40, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            y_position += 45

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total = sum(counts.values())

        cv2.putText(
            annotated_frame,
            f"TOTAL : {total}",
            (40, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        annotated_frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # F = fullscreen toggle
    elif key == ord("f"):

        fullscreen = not fullscreen

        if fullscreen:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

        else:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

            cv2.resizeWindow(
                WINDOW_NAME,
                1280,
                720
            )

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Webcam stopped.")