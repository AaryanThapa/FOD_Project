import cv2
import mediapipe as mp
import pygame
import pymunk
import pymunk.pygame_util
import random
import math
import time

# ---------- Pygame & Pymunk Setup ----------
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Finger Bridge + Bouncy Balls (with Sparks + Score)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

space = pymunk.Space()
# NOTE: to keep your original feel, gravity remains positive (we flip y for drawing)
space.gravity = (0, 900)

balls = []            # list of pymunk.Circle shapes
bridge_segment = None
bridge_exists = False

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
last_ball_time = 0

score = 0

# ---------- Particles (sparks) ----------
class Spark:
    def __init__(self, pos, vel, life, color):
        # pos given in pymunk coords (same as shape.body.position)
        self.x = pos[0]
        self.y = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.life = life          # seconds remaining
        self.max_life = life
        self.color = color        # (r,g,b)
        self.size = random.randint(2, 4)

    def update(self, dt):
        # gravity effect (in pymunk coordinate convention used)
        self.vy += space.gravity[1] * dt * 0.25   # scaled gravity for sparks
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    def draw(self, surf):
        # convert pymunk coords to pygame screen coords
        sx = int(self.x)
        sy = int(HEIGHT - self.y)
        # fade factor
        t = max(0.0, min(1.0, self.life / self.max_life))
        # color fade to black
        col = (int(self.color[0] * t), int(self.color[1] * t), int(self.color[2] * t))
        # draw as short line to give spark streak
        end_x = int(sx + self.vx * 0.02 * (1 + (1 - t) * 3))
        end_y = int(sy - self.vy * 0.02 * (1 + (1 - t) * 3))
        try:
            pygame.draw.line(surf, col, (sx, sy), (end_x, end_y), max(1, self.size // 2))
        except Exception:
            pass

sparks = []

# ---------- Utility: Fire palette generator ----------
def fire_color():
    # returns a (r,g,b) tuple reminiscent of fire (yellow -> orange -> red)
    stage = random.random()
    if stage < 0.4:
        # yellowish
        r = random.randint(200, 255)
        g = random.randint(150, 200)
        b = random.randint(20, 60)
    elif stage < 0.8:
        # orange
        r = random.randint(200, 255)
        g = random.randint(90, 150)
        b = random.randint(10, 40)
    else:
        # red
        r = random.randint(160, 230)
        g = random.randint(20, 80)
        b = random.randint(0, 30)
    return (r, g, b)

# ---------- Ball creator ----------
def create_ball():
    body = pymunk.Body()
    # spawn x inside the central region
    body.position = (random.randint(200, 600), 50)
    shape = pymunk.Circle(body, 18)
    shape.mass = 1
    shape.elasticity = 0.8
    shape.friction = 0.8
    # keep a visible color attribute
    shape.color = (random.randint(50,255), random.randint(50,255), random.randint(50,255), 255)
    # collision_type used in original design (we won't rely on pymunk handlers here)
    shape.collision_type = 2  # balls (kept for clarity)
    # custom flags
    shape._has_exploded = False
    shape._touched_bridge = False
    space.add(body, shape)
    return shape

# We'll collect shapes to remove after stepping physics
balls_to_remove = []

# ---------- Spark spawning ----------
def spawn_sparks(ball_shape):
    # spawn multiple sparks at ball position
    body = ball_shape.body
    pos = (body.position.x, body.position.y)
    base_color = fire_color()  # fire palette
    # spawn many fast short-lived particles
    n = random.randint(10, 18)
    for i in range(n):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(120, 420)   # faster for fireworks feel
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        life = random.uniform(0.35, 0.9)   # short-lived sparks
        c = base_color
        sparks.append(Spark(pos, (vx, vy), life, c))

# ---------- Helper: distance from point to segment ----------
def point_segment_distance(px, py, ax, ay, bx, by):
    """Return distance and closest point from P(px,py) to segment A(ax,ay)-B(bx,by)."""
    # vector AB
    abx = bx - ax
    aby = by - ay
    ab_len2 = abx*abx + aby*aby
    if ab_len2 == 0:
        # A and B are the same point
        dx = px - ax
        dy = py - ay
        return math.hypot(dx, dy), (ax, ay)
    # project AP onto AB, compute t in [0,1]
    apx = px - ax
    apy = py - ay
    t = (apx*abx + apy*aby) / ab_len2
    if t < 0:
        closest = (ax, ay)
    elif t > 1:
        closest = (bx, by)
    else:
        closest = (ax + abx * t, ay + aby * t)
    dx = px - closest[0]
    dy = py - closest[1]
    return math.hypot(dx, dy), closest

# ---------- Main Loop ----------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Read camera ---
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- Remove old bridge segment if exists ---
    if bridge_segment and bridge_segment in space.shapes:
        try:
            space.remove(bridge_segment)
        except Exception:
            pass
    bridge_segment = None
    bridge_exists = False

    # --- Create bridge from thumb (4) to index (8) if hand detected ---
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        # Map normalized MediaPipe coords to Pygame window coords (x across width, y across height)
        x1 = int(lm[4].x * WIDTH)
        y1 = int(lm[4].y * HEIGHT)
        x2 = int(lm[8].x * WIDTH)
        y2 = int(lm[8].y * HEIGHT)

        # Create physics segment attached to static_body
        # Use the same collision_type (1) in original design (kept for clarity)
        segment = pymunk.Segment(space.static_body, (x1, y1), (x2, y2), 12)
        segment.elasticity = 0.7
        segment.friction = 0.9
        segment.collision_type = 1   # bridge (kept for clarity)
        # store for removal and drawing
        space.add(segment)
        bridge_segment = segment
        bridge_exists = True

        # Draw landmarks on webcam preview
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # --- Spawn balls every 1.5 seconds ---
    now = pygame.time.get_ticks()
    if now - last_ball_time > 1500:
        balls.append(create_ball())
        last_ball_time = now

    # --- Physics step ---
    dt = 1.0 / 60.0
    space.step(dt)

    # --- Manual collision detection: ball <-> bridge segment ---
    # We do this after stepping physics. This avoids depending on pymunk collision handler API.
    if bridge_exists and bridge_segment is not None:
        # segment endpoints in pymunk coords:
        ax, ay = bridge_segment.a.x, bridge_segment.a.y
        bx, by = bridge_segment.b.x, bridge_segment.b.y
        bridge_radius = bridge_segment.radius if hasattr(bridge_segment, "radius") else 12

        for ball in balls[:]:
            if getattr(ball, "_has_exploded", False):
                continue
            # ball center
            px, py = ball.body.position.x, ball.body.position.y
            dist, closest = point_segment_distance(px, py, ax, ay, bx, by)
            # ball radius (circle radius used when created)
            ball_radius = ball.radius if hasattr(ball, "radius") else 18
            if dist <= (ball_radius + bridge_radius):
                # Collision detected -> trigger explosion + color change + mark removal + score
                ball._touched_bridge = True
                ball.color = (255, 255, 200, 255)
                spawn_sparks(ball)
                ball._has_exploded = True
                balls_to_remove.append(ball)
                score += 1
                # continue to next ball

    # --- Remove any balls that were marked for removal (exploded) ---
    if balls_to_remove:
        for b in balls_to_remove:
            try:
                if b in balls:
                    balls.remove(b)
                if b.body in space.bodies:
                    space.remove(b.body, b)
            except Exception:
                pass
        balls_to_remove.clear()

    # --- Update sparks ---
    dt_seconds = clock.get_time() / 1000.0 if clock.get_time() > 0 else dt
    # clamp dt_seconds to reasonable range
    if dt_seconds <= 0 or dt_seconds > 0.1:
        dt_seconds = dt
    for s in sparks[:]:
        s.update(dt_seconds)
        if s.life <= 0:
            sparks.remove(s)

    # --- Draw everything in Pygame ---
    screen.fill((20, 20, 40))

    # Draw balls
    for ball in balls[:]:
        # remove balls that fall far below the screen
        if ball.body.position.y > HEIGHT + 100:
            # ball lost -> penalty
            try:
                if ball in balls:
                    balls.remove(ball)
                if ball.body in space.bodies:
                    space.remove(ball.body, ball)
            except Exception:
                pass
            score -= 1
            continue

        # draw using stored color (RGBA) -> drop alpha for pygame
        col = (ball.color[0], ball.color[1], ball.color[2])
        pos = int(ball.body.position.x), int(HEIGHT - ball.body.position.y)
        pygame.draw.circle(screen, col, pos, 18)

    # Draw sparks (fire effect)
    for s in sparks:
        s.draw(screen)

    # Draw bridge
    if bridge_exists and bridge_segment:
        # pymunk segment endpoints:
        p1 = int(bridge_segment.a.x), int(HEIGHT - bridge_segment.a.y)
        p2 = int(bridge_segment.b.x), int(HEIGHT - bridge_segment.b.y)
        pygame.draw.line(screen, (200, 120, 30), p1, p2, 28)  # warm tone for bridge
        pygame.draw.circle(screen, (255, 200, 80), p1, 18)
        pygame.draw.circle(screen, (255, 200, 80), p2, 18)

    # Draw score
    score_surf = font.render(f"Score: {score}", True, (240, 240, 240))
    screen.blit(score_surf, (12, 12))

    pygame.display.flip()
    clock.tick(60)

    # --- Show webcam feed (optional) ---
    cv2.imshow("Your Hand", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pygame.quit()
