import cv2
import subprocess
from ultralytics import YOLO

# Load YOLO model
model_path = r"C:\Users\My project\collegeA\test\best.pt"  # Update this with your model path
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)
current_process = None  # Track running game

detected_object = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Detect object once
            if label.lower() in ["square", "grid", "maze"]:
                detected_object = label.lower()
                break

    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit detection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detection closed.")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

    if detected_object:
        break  # Stop loop once an object is detected

# Close webcam feed
cap.release()
cv2.destroyAllWindows()

# Launch the detected game
if detected_object:
    game_script = None
    if detected_object == "square":
        print("Launching Tetris...")
        game_script = "tetris.py"
    elif detected_object == "grid":
        print("Launching Snake...")
        game_script = "snake.py"
    elif detected_object == "maze":
        print("Launching Maze Game...")
        game_script = "maze.py"

    if game_script:
        current_process = subprocess.Popen(["python", game_script])

        # Wait for 'q' to close the game
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing game...")
                current_process.terminate()
                current_process.wait()
                break

        print("Game closed successfully.")
