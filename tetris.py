import cv2
import numpy as np
import time

# Video capture
cap = cv2.VideoCapture(0)

# Grid properties
grid_size = 450
cell_size = 50
rows, cols = grid_size // cell_size, grid_size // cell_size  # 9x9 grid

# Tetrimino shapes (as 2D arrays)
TETROMINOS = [
    np.array([[1, 1, 1], [0, 1, 0]]),  # T-shape
    np.array([[1, 1, 1, 1]]),  # I-shape
    np.array([[1, 1], [1, 1]]),  # O-shape
    np.array([[0, 1, 1], [1, 1, 0]]),  # S-shape
    np.array([[1, 1, 0], [0, 1, 1]]),  # Z-shape
    np.array([[1, 1, 1], [1, 0, 0]]),  # L-shape
    np.array([[1, 1, 1], [0, 0, 1]])   # J-shape
]

# Initialize block
def new_block():
    shape = TETROMINOS[np.random.randint(len(TETROMINOS))]
    return {"shape": shape, "x": cols // 2 - len(shape[0]) // 2, "y": 0, "speed": 1}

def reset_game():
    global grid, block, score
    grid = np.zeros((rows, cols), dtype=int)
    block = new_block()
    score = 0  # Reset score

reset_game()

def is_valid_position(block, x_offset=0, y_offset=0):
    """Check if the block can be placed at the new position."""
    for y in range(block["shape"].shape[0]):
        for x in range(block["shape"].shape[1]):
            if block["shape"][y, x]:  # If cell is occupied
                new_x = block["x"] + x + x_offset
                new_y = block["y"] + y + y_offset
                if new_x < 0 or new_x >= cols or new_y >= rows or grid[new_y, new_x]:
                    return False
    return True

def place_block(block):
    """Set block into the grid and check for row completion."""
    global score
    for y in range(block["shape"].shape[0]):
        for x in range(block["shape"].shape[1]):
            if block["shape"][y, x]:
                grid[block["y"] + y, block["x"] + x] = 1
    
    # Check for full rows
    full_rows = [r for r in range(rows) if all(grid[r])]
    for r in full_rows:
        grid[1:r+1] = grid[:r]  # Shift everything down
        grid[0] = 0  # Clear the top row
        score += 10  # Increment score for each cleared row

# Handle user input
key_pressed = None

def key_callback(event):
    global key_pressed
    key_pressed = chr(event & 0xFF) if event != -1 else None

cv2.namedWindow("Webcam Feed")
cv2.setWindowProperty("Webcam Feed", cv2.WND_PROP_TOPMOST, 1)

# Game loop
last_time = time.time()
game_over = False

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
        cv2.drawContours(frame, [best_square], 0, (0, 255, 0), 2)

        pts = np.array([point[0] for point in best_square], dtype=np.float32)
        pts = sorted(pts, key=lambda x: (x[1], x[0]))
        top_left, top_right = sorted(pts[:2], key=lambda x: x[0])
        bottom_left, bottom_right = sorted(pts[2:], key=lambda x: x[0])

        dst_pts = np.float32([[0, 0], [grid_size, 0], [0, grid_size], [grid_size, grid_size]])
        matrix = cv2.getPerspectiveTransform(
            np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32),
            dst_pts
        )
        warped = cv2.warpPerspective(frame, matrix, (grid_size, grid_size))

        if not game_over:
            # Move block every 0.5 seconds
            if time.time() - last_time > 0.5:
                if is_valid_position(block, y_offset=1):
                    block["y"] += 1
                else:
                    place_block(block)
                    block = new_block()
                    if not is_valid_position(block):  # Check if game over
                        game_over = True
                last_time = time.time()

            # Handle user input
            key_callback(cv2.waitKey(1))
            if key_pressed == "a" and is_valid_position(block, x_offset=-1):
                block["x"] -= 1
            elif key_pressed == "d" and is_valid_position(block, x_offset=1):
                block["x"] += 1
            elif key_pressed == "s" and is_valid_position(block, y_offset=1):
                block["y"] += 1
            elif key_pressed == "w":  # Rotate
                rotated = np.rot90(block["shape"])
                if is_valid_position({"shape": rotated, "x": block["x"], "y": block["y"]}):
                    block["shape"] = rotated

        # Draw placed blocks
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]:
                    cv2.rectangle(warped, (c * cell_size, r * cell_size),
                                  (c * cell_size + cell_size, r * cell_size + cell_size),
                                  (0, 0, 255), thickness=cv2.FILLED)

        # Draw falling block
        if not game_over:
            for y in range(block["shape"].shape[0]):
                for x in range(block["shape"].shape[1]):
                    if block["shape"][y, x]:
                        cv2.rectangle(warped, ((block["x"] + x) * cell_size, (block["y"] + y) * cell_size),
                                      ((block["x"] + x) * cell_size + cell_size, (block["y"] + y) * cell_size + cell_size),
                                      (255, 0, 0), thickness=cv2.FILLED)

        inv_matrix = cv2.getPerspectiveTransform(dst_pts, np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32))
        restored = cv2.warpPerspective(warped, inv_matrix, (frame.shape[1], frame.shape[0]))

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [best_square], (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, mask_inv)
        grid_fg = cv2.bitwise_and(restored, mask)
        frame = cv2.add(frame_bg, grid_fg)

    # Display score
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        cv2.putText(frame, "Game Over! Press 'R' to Restart", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        key_callback(cv2.waitKey(1))
    if key_pressed == "q":
        print("Closing game...")
        break
    elif key_pressed == "r" and game_over:
        reset_game()
        
    cv2.imshow("Webcam Feed", frame)

cap.release()
cv2.destroyAllWindows()