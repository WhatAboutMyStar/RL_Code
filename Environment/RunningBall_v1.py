import pygame
import random
import numpy as np
import cv2

class RunningBall:
    def __init__(self, fps=50):
        pygame.init()
        self.fps = fps
        self.fcclock = pygame.time.Clock()

        self.size = (228, 228)
        self.enemy_x = random.randint(120, 200)
        self.enemy_y = random.randint(120, 200)

        self.x = random.randint(20, 80)
        self.y = random.randint(20, 80)
        self.done = False

        self.screen = pygame.display.set_mode(self.size)

    def step(self, action):
        self.updateWithInput(action)
        self.updateWithoutInput()
        distance = self.distance(self.x, self.y, self.enemy_x, self.enemy_y)
        next_state = [self.x, self.y, self.enemy_x, self.enemy_y, distance, 20, 40, 228, 228]
        next_state = np.array(next_state)

        if self.done:
            reward = -100
        else:
            reward = 1
        info = {}

        return next_state, reward, self.done, info


    def reset(self):
        self.enemy_x = random.randint(120, 200)
        self.enemy_y = random.randint(120, 200)

        self.x = random.randint(20, 80)
        self.y = random.randint(20, 80)
        self.done = False
        pygame.draw.circle(surface=self.screen, color=[255, 0, 0], center=[self.x, self.y], radius=20, width=0)
        pygame.draw.circle(surface=self.screen, color=[0, 255, 0], center=[self.enemy_x, self.enemy_y], radius=20,
                           width=0)

        distance = self.distance(self.x, self.y, self.enemy_x, self.enemy_y)
        next_state = [self.x, self.y, self.enemy_x, self.enemy_y, distance, 20, 40, 228, 228]
        next_state = np.array(next_state)

        return next_state

    def updateWithInput(self, action):
        if action == 0: # w
            self.y = self.y - 5
            if self.y < 20:
                self.y = 20
        elif action == 1: # s
            self.y = self.y + 5
            if self.y > 208:
                self.y = 208
        elif action == 2: # a
            self.x = self.x - 5
            if self.x < 20:
                self.x = 20
        elif action == 3: # d
            self.x = self.x + 5
            if self.x > 208:
                self.x = 208

    def updateWithoutInput(self):
        if self.enemy_x < self.x:
            self.enemy_x += 1
        elif self.enemy_x > self.x:
            self.enemy_x -= 1

        if self.enemy_y < self.y:
            self.enemy_y += 1
        elif self.enemy_y > self.y:
            self.enemy_y -= 1

        if self.distance(self.x, self.y, self.enemy_x, self.enemy_y) < 40:
            self.done = True

    def clearScreen(self):
        self.screen.fill([255, 255, 255])

    def distance(self, x1, y1, x2, y2):
        return pow((x1 - x2)**2 + (y1 - y2)**2, 0.5)

    def close(self):
        self.done = True
        pygame.quit()

    def render(self):
        self.clearScreen()
        pygame.draw.circle(surface=self.screen, color=[255, 0, 0], center=[self.x, self.y], radius=20, width=0)
        pygame.draw.circle(surface=self.screen, color=[0, 255, 0], center=[self.enemy_x, self.enemy_y], radius=20,
                           width=0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.fcclock.tick(self.fps)
        pygame.display.flip()

    def sample(self):
        return random.randint(0, 4)

    def __repr__(self):
        return "RunningBall()"

if __name__ == '__main__':
    env = RunningBall()
    total_reward = 0
    while True:
        env.render()
        action = env.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(total_reward)
            env.reset()


