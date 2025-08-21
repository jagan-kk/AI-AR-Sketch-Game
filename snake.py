import cv2
import numpy as np
import time
import random
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Grid properties
grid_size = 450
cell_size = 50
rows, cols = grid_size // cell_size, grid_size // cell_size  # 9x9 grid

# Snake properties
snake = [(rows // 2, cols // 2)]  # Starting position of the snake (center)
snake_direction = (0, 1)  # Moving right
snake_length = 1

# Food properties
food = None

# Score
score = 0

# Handle user input
key_pressed = None

def key_callback(event):
    global key_pressed
    key_pressed = chr(event & 0xFF) if event != -1 else None

cv2.namedWindow("Webcam Feed")
cv2.setWindowProperty("Webcam Feed", cv2.WND_PROP_TOPMOST, 1)

# Generate food at a random location
def generate_food():
    while True:
        food_x = random.randint(0, cols - 1)
        food_y = random.randint(0, rows - 1)
        if (food_y, food_x) not in snake:  # Ensure food doesn't spawn on the snake
            return (food_y, food_x)

food = generate_food()

# Reset game state
def reset_game():
    global snake, snake_direction, snake_length, food, score, game_over
    snake = [(rows // 2, cols // 2)]  # Reset snake to center
    snake_direction = (0, 1)  # Reset direction
    snake_length = 1  # Reset length
    food = generate_food()  # Generate new food
    score = 0  # Reset score
    game_over = False  # Reset game over state

reset_game()

# Game loop
last_time = time.time()
move_interval = 0.4
next_move_time = time.time() + move_interval

# Store the last valid square detection
last_valid_square = None

# Hand gesture control
def detect_hand_gesture(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the index finger tip (landmark 8) and wrist (landmark 0)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            # Calculate the direction vector from wrist to index finger tip
            dx = index_x - wrist_x
            dy = index_y - wrist_y

            # Determine the direction based on the angle of the direction vector
            angle = np.arctan2(dy, dx) * 180 / np.pi

            if -45 <= angle < 45:
                direction = "Right"
            elif 45 <= angle < 135:
                direction = "Down"
            elif -135 <= angle < -45:
                direction = "Up"
            else:
                direction = "Left"

            # Draw the direction on the frame
            cv2.putText(frame, f"Direction: {direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Return the direction
            return direction
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_square = None
    max_area = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = w * h
            if 0.9 <= aspect_ratio <= 1.1 and area > max_area:
                best_square = approx
                max_area = area

    if best_square is not None and max_area > 5000:
        last_valid_square = best_square

    if last_valid_square is not None:
        cv2.drawContours(frame, [last_valid_square], 0, (0, 255, 0), 2)

        pts = np.array([point[0] for point in last_valid_square], dtype=np.float32)
        pts = sorted(pts, key=lambda x: (x[1], x[0]))
        top_left, top_right = sorted(pts[:2], key=lambda x: x[0])
        bottom_left, bottom_right = sorted(pts[2:], key=lambda x: x[0])

        dst_pts = np.float32([[0, 0], [grid_size, 0], [0, grid_size], [grid_size, grid_size]])
        matrix = cv2.getPerspectiveTransform(
            np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32),
            dst_pts
        )
        warped = cv2.warpPerspective(frame, matrix, (grid_size, grid_size))

        # Detect hand gesture and direction
        direction = detect_hand_gesture(frame)
        if direction:
            if direction == "Left":
                snake_direction = (0, -1)
            elif direction == "Right":
                snake_direction = (0, 1)
            elif direction == "Up":
                snake_direction = (-1, 0)
            elif direction == "Down":
                snake_direction = (1, 0)

        current_time = time.time()
        if current_time >= next_move_time and not game_over:
            head_y, head_x = snake[0]
            new_head = (head_y + snake_direction[0], head_x + snake_direction[1])

            # Wrap around the grid
            new_head = (new_head[0] % rows, new_head[1] % cols)

            if new_head in snake:
                game_over = True  # Game over if snake collides with itself
            else:
                snake.insert(0, new_head)

                if new_head == food:
                    snake_length += 1
                    score += 10  # Increase score when food is eaten
                    food = generate_food()
                else:
                    snake.pop()

            next_move_time = current_time + move_interval

        for y, x in snake:
            cv2.rectangle(warped, (x * cell_size, y * cell_size), 
                          (x * cell_size + cell_size, y * cell_size + cell_size), 
                          (0, 255, 0), thickness=cv2.FILLED)

        food_y, food_x = food
        cv2.rectangle(warped, (food_x * cell_size, food_y * cell_size),
                      (food_x * cell_size + cell_size, food_y * cell_size + cell_size),
                      (0, 0, 255), thickness=cv2.FILLED)

        inv_matrix = cv2.getPerspectiveTransform(dst_pts, np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32))
        restored = cv2.warpPerspective(warped, inv_matrix, (frame.shape[1], frame.shape[0]))

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [last_valid_square], (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, mask_inv)
        grid_fg = cv2.bitwise_and(restored, mask)
        frame = cv2.add(frame_bg, grid_fg)

    # Display score
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Game over message and restart option
    if game_over:
        cv2.putText(frame, "Game Over! Press 'R' to Restart", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        key_callback(cv2.waitKey(1))
        if key_pressed == "r":
            reset_game()

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()