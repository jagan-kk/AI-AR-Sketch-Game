import cv2
import numpy as np

# Video capture
cap = cv2.VideoCapture(0)

# Ball properties
ball_radius = 4
ball_position = None  
ball_velocity = np.array([0, 0], dtype=np.float32)
gravity = np.array([0, 0.7], dtype=np.float32)  # Stronger gravity
friction = 0.98
jump_strength = -12  # Higher jump strength

# Store detected square and lines
confirmed_square = None  
detected_lines = []
previous_square_center = None  

def detect_largest_square(frame):
    """Detects and returns the largest square in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_square = None
    max_area = 0
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = w * h
            
            if 0.9 <= aspect_ratio <= 1.1 and area > max_area:
                max_area = area
                largest_square = approx
    
    return largest_square

def detect_lines_in_square(frame, square):
    """Detects and stores rigid lines within the given square, ensuring stability."""
    global detected_lines
    
    if square is None:
        return  

    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [square], 0, (255, 255, 255), thickness=cv2.FILLED)
    masked_frame = cv2.bitwise_and(frame, mask)
    
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)

    if lines is not None:
        new_detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            new_detected_lines.append(((x1, y1), (x2, y2)))

        # Only update detected lines if there's a significant change
        if len(new_detected_lines) > len(detected_lines) * 0.5:
            detected_lines = new_detected_lines  

def spawn_ball():
    """Spawns the ball inside the confirmed square on the lowest rigid line."""
    global ball_position
    if confirmed_square is None or not detected_lines:
        return  

    # Filter out only horizontal lines
    horizontal_lines = [(x1, y1, x2, y2) for (x1, y1), (x2, y2) in detected_lines if abs(y1 - y2) < 5]

    if not horizontal_lines:
        print("No horizontal lines detected! Cannot spawn ball.")
        return  # Exit function safely if no valid horizontal lines are found

    # Find the lowest horizontal line
    lowest_line_y = max(y1 for _, y1, _, _ in horizontal_lines)

    # Place the ball in the center above the lowest detected line
    ball_x = (horizontal_lines[0][0] + horizontal_lines[0][2]) // 2  
    ball_position = np.array([ball_x, lowest_line_y - ball_radius - 2], dtype=np.float32)
    print(f"Ball spawned at ({ball_position[0]}, {ball_position[1]})")


def check_collision():
    """Handles ball collision with detected lines for both horizontal and vertical rigidity."""
    global ball_position, ball_velocity

    if detected_lines is None or ball_position is None:
        return  

    for line in detected_lines:
        (x1, y1), (x2, y2) = line

        # Horizontal Collision (Land on top)
        if min(x1, x2) <= ball_position[0] <= max(x1, x2) and abs(ball_position[1] + ball_radius - y1) < 5:
            if ball_velocity[1] > 0:  # Falling down
                ball_velocity[1] = 0  
                ball_position[1] = y1 - ball_radius - 1  
            return  

        # Vertical Collision (Prevent passing through walls)
        if min(y1, y2) <= ball_position[1] <= max(y1, y2) and abs(ball_position[0] - x1) < 5:
            if ball_velocity[0] > 0:  # Moving right
                ball_position[0] = x1 - ball_radius - 1  
            elif ball_velocity[0] < 0:  # Moving left
                ball_position[0] = x1 + ball_radius + 1  
            ball_velocity[0] = 0  # Stop movement
            return  

def update_ball():
    """Updates ball movement based on physics and square movement."""
    global ball_position, ball_velocity, previous_square_center
    if ball_position is None:
        return  

    # Track square movement and move the ball with it
    if confirmed_square is not None:
        current_square_center = np.mean(confirmed_square, axis=0)[0]
        if previous_square_center is not None:
            displacement = current_square_center - previous_square_center
            if check_if_ball_on_square():
                ball_position += displacement  # Move ball only if it was on square
        previous_square_center = current_square_center

    # Apply physics
    ball_velocity += gravity  
    ball_velocity *= friction  
    ball_position += ball_velocity  

    check_collision()

    # Keep ball within screen limits
    ball_position[0] = max(ball_radius, min(640 - ball_radius, ball_position[0]))
    ball_position[1] = max(ball_radius, min(480 - ball_radius, ball_position[1]))

def check_if_ball_on_square():
    """Returns True if the ball is sitting on the confirmed square."""
    if confirmed_square is None or ball_position is None:
        return False

    x, y, w, h = cv2.boundingRect(confirmed_square)
    return x <= ball_position[0] <= x + w and abs(ball_position[1] - (y + h)) < 5


def handle_input(key):
    """Handles user keyboard input for jumping and moving."""
    global ball_velocity

    if key == 32:  # Space bar for jumping
        # Allow jumping if the ball is touching a horizontal surface
        for (x1, y1), (x2, y2) in detected_lines:
            if min(x1, x2) <= ball_position[0] <= max(x1, x2) and abs(ball_position[1] + ball_radius - y1) < 5:
                ball_velocity[1] = jump_strength  # Jumping now works properly
                return  

    elif key == ord('a'):
        ball_velocity[0] -= 2  # Move left
    elif key == ord('d'):
        ball_velocity[0] += 2  # Move right
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect the largest square continuously
    largest_square = detect_largest_square(frame)
    if largest_square is not None:
        # Only update square if it's significantly different
        if confirmed_square is None or cv2.norm(confirmed_square - largest_square) > 10:
            confirmed_square = largest_square
            detect_lines_in_square(frame, confirmed_square)

        cv2.drawContours(frame, [confirmed_square], 0, (0, 255, 0), 2)
        
        if ball_position is None:  
            spawn_ball()
    
    update_ball()
    
    if ball_position is not None:
        cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), ball_radius, (255, 0, 0), -1)
    
    cv2.imshow("Detected Square and Ball", frame)
    
    key = cv2.waitKey(1) & 0xFF
    handle_input(key)
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
