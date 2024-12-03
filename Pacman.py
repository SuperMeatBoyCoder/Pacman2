import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
from random import randrange
import random
import copy
import os
import time
from collections import deque

SheetsPath = "Assets/SpriteSheets/"
TextPath = "Assets/TextImages/"
DataPath = "Assets/Data/"
MusicPath = "Assets/Music/"

pygame.mixer.init()
pygame.init()
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

# 28 Across 31 Tall 1: Empty Space 2: Tic-Tak 3: Wall 4: Ghost safe-space 5: Special Tic-Tak
originalGameBoard = [
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    [3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3],
    [3, 6, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 6, 3],
    [3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3],
    [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    [3, 2, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 3],
    [3, 2, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 2, 3],
    [3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 3, 4, 4, 4, 4, 4, 4, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],  # Middle Lane Row: 14
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 3, 4, 4, 4, 4, 4, 4, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    [3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3],
    [3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3],
    [3, 6, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 6, 3],
    [3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3],
    [3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3],
    [3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3],
    [3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3],
    [3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3],
    [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
]
gameBoard = copy.deepcopy(originalGameBoard)
spriteRatio = 3 / 2
square = 20  # Size of each unit square
FPS = 30
spriteOffset = square * (1 - spriteRatio) * (1 / 2)
(width, height) = (len(gameBoard[0]) * square, len(gameBoard) * square)  # Game screen
screen = pygame.display.set_mode((width, height))
pygame.display.flip()
pelletColor = (222, 161, 133)
movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]
INF = int(1e9)

PLAYING_KEYS = {
    "up": [pygame.K_w, pygame.K_UP],
    "down": [pygame.K_s, pygame.K_DOWN],
    "right": [pygame.K_d, pygame.K_RIGHT],
    "left": [pygame.K_a, pygame.K_LEFT]
}

boardImage = pygame.image.load(SheetsPath + "GameBoardSheet.png")
boardImage = pygame.transform.scale(boardImage, (width, height - 5 * square))
elementSheet = pygame.image.load(SheetsPath + "ElementSheet.png")
elementSheet = pygame.transform.scale(elementSheet, (16 * square, 10 * square))
charsImages = {}
for filename in os.listdir(TextPath):
    gChar = os.path.splitext(filename)[0]
    charsImages[gChar] = pygame.image.load(f"{TextPath}{filename}")
    charsImages[gChar] = pygame.transform.scale(charsImages[gChar], (square, square))


def loadElement(tile_num):
    return elementSheet.subsurface((tile_num % 16 * square, tile_num // 16 * square, square, square))


def drawReady():
    for i, char in enumerate("READY!"):
        char = "O" + char
        letter = charsImages[char]
        screen.blit(letter, ((11 + i) * square, 20 * square, square, square))


def calcDistance(a, b):
    dR = a[0] - b[0]
    dC = a[1] - b[1]
    return math.hypot(dR, dC)


class Game:
    def __init__(self, level, score):
        self.paused = True
        self.tictakChangeDelay = 10
        self.tictakChangeCount = 0
        self.highScore = self.getHighScore()
        self.score = score
        self.level = level
        self.lives = 3
        self.ghosts = [Ghost(14.0, 13.5, "red", 0), Ghost(17.0, 11.5, "blue", 1), Ghost(17.0, 13.5, "pink", 2),
                       Ghost(17.0, 15.5, "orange", 3)]
        self.pacman = Pacman(26.0, 13.5)  # Center of Second Last Row
        self.total = self.getCount()
        self.ghostScore = 200
        self.levels = [[350, 250], [150, 450], [150, 450], [0, 600]]
        random.shuffle(self.levels)
        # Level index and Level Progress
        self.ghostStates = [[1, 0], [0, 0], [1, 0], [0, 0]]
        for i, state in enumerate(self.ghostStates):
            state[0] = randrange(2)
            state[1] = randrange(self.levels[i][state[0]] + 1)
        self.collected = 0
        self.started = False
        self.points = []
        self.pointsTimer = 10
        # Berry Spawn Time, Berry Death Time, Berry Eaten
        self.berryState = [200, 400, False]
        self.berryLocation = [20.0, 13.5]
        self.berries = list(range(80, 88))
        self.berriesCollected = []
        self.levelTimer = 0
        self.berryScore = 100
        self.lockedInTimer = 100
        self.lockedIn = True
        self.extraLifeGiven = False
        self.fingers = [(0, 0), (INF, INF)]
        self.gameOver = False

    # Driver method: The games primary update method
    def update(self):

        self.tictakChangeCount += 1

        if self.tictakChangeCount == self.tictakChangeDelay:
            # Changes the color of special Tic-Taks
            self.flipColor()
            self.tictakChangeCount = 0

        if frames % 2 == 0 and self.getTarget():
            self.paused = False
            self.started = True

        if self.paused or not self.started:
            self.render()
            pygame.display.update()
            return

        self.levelTimer += 1

        if self.score >= 10000 and not self.extraLifeGiven:
            self.lives += 1
            self.extraLifeGiven = True
            self.forcePlayMusic("pacman_extrapac.wav")

        # Check if the ghost should chase pacman
        for i, state in enumerate(self.ghostStates):
            state[1] += 1
            if state[1] >= self.levels[i][state[0]]:
                state[1] = 0
                state[0] = not state[0]

        for i, ghost in enumerate(self.ghosts):
            if not ghost.attacked and not ghost.dead and self.ghostStates[i][0] == 0:
                ghost.target = (self.pacman.row, self.pacman.col)

        if self.levelTimer == self.lockedInTimer:
            self.lockedIn = False

        for ghost in self.ghosts:
            ghost.update()

        self.pacman.update()
        self.pacman.col %= len(gameBoard[0])
        if self.pacman.row % 1.0 == 0 and self.pacman.col % 1.0 == 0:
            if gameBoard[int(self.pacman.row)][int(self.pacman.col)] == 2:
                self.playMusic("munch_1.wav")
                gameBoard[int(self.pacman.row)][int(self.pacman.col)] = 1
                self.score += 10
                self.collected += 1
                # Fill tile with black
                pygame.draw.rect(screen, (0, 0, 0),
                                 (self.pacman.col * square, self.pacman.row * square, square, square))
            elif gameBoard[int(self.pacman.row)][int(self.pacman.col)] == 5 or \
                    gameBoard[int(self.pacman.row)][int(self.pacman.col)] == 6:
                self.forcePlayMusic("power_pellet.wav")
                gameBoard[int(self.pacman.row)][int(self.pacman.col)] = 1
                self.collected += 1
                # Fill tile with black
                pygame.draw.rect(screen, (0, 0, 0),
                                 (self.pacman.col * square, self.pacman.row * square, square, square))
                self.score += 50
                self.ghostScore = 200
                for ghost in self.ghosts:
                    ghost.attackedCount = 0
                    ghost.attacked = True
                    ghost.setTarget()
        self.checkSurroundings()
        self.highScore = max(self.score, self.highScore)

        global running
        if self.collected == self.total:
            print("New Level")
            self.forcePlayMusic("intermission.wav")
            self.level += 1
            self.newLevel()

        if self.level == 9:
            print("You win", self.level, len(self.levels))
            running = False
        self.render()
        pygame.display.update()

    # Render method
    def render(self):
        screen.fill((0, 0, 0))  # Flushes the screen
        # Draws game elements
        currentTile = 0
        for i in range(3, len(gameBoard) - 2):
            for j in range(len(gameBoard[0])):
                if gameBoard[i][j] == 3:  # Draw wall

                    # Display image of tile
                    screen.blit(boardImage, (j * square, i * square), (j * square, (i - 3) * square, square, square))

                elif gameBoard[i][j] == 2:  # Draw Tic-Tak
                    pygame.draw.circle(screen, pelletColor, (j * square + square // 2, i * square + square // 2),
                                       square // 4)
                elif gameBoard[i][j] == 5:  # Black Special Tic-Tak
                    pygame.draw.circle(screen, (0, 0, 0), (j * square + square // 2, i * square + square // 2),
                                       square // 2)
                elif gameBoard[i][j] == 6:  # White Special Tic-Tak
                    pygame.draw.circle(screen, pelletColor, (j * square + square // 2, i * square + square // 2),
                                       square // 2)

                currentTile += 1

        pointsToDraw = []
        for point in self.points:
            if point[3] < self.pointsTimer:
                pointsToDraw.append([point[2], point[0], point[1]])
                point[3] += 1
            else:
                self.points.remove(point)

        for point in pointsToDraw:
            self.drawPoints(point[0], point[1], point[2])

        # Draw Sprites
        for ghost in self.ghosts:
            ghost.draw(not game.paused)
        self.displayScore()
        self.displayBerries()
        self.displayLives()
        self.drawBerry()
        if not self.gameOver:
            self.pacman.draw()
            self.displayFingers()
        if self.paused:
            drawReady()

    @staticmethod
    def playMusic(music):
        # return False # Uncomment to disable music
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.unload()
            pygame.mixer.music.load(MusicPath + music)
            pygame.mixer.music.queue(MusicPath + music)
            pygame.mixer.music.play()

    @staticmethod
    def forcePlayMusic(music):
        # return False # Uncomment to disable music
        pygame.mixer.music.unload()
        pygame.mixer.music.load(MusicPath + music)
        pygame.mixer.music.play()

    def checkSurroundings(self):
        # Check if pacman got killed
        for ghost in self.ghosts:
            if self.touchingPacman(ghost.row, ghost.col) and not ghost.attacked:
                if self.lives == 1:
                    print("You lose")
                    self.forcePlayMusic("death_1.wav")
                    # Removes the ghosts from the screen
                    self.ghosts = []
                    self.pacman.draw()
                    pygame.display.update()
                    time.sleep(0.4)
                    self.gameOver = True
                    for i in range(13):
                        self.gameOverFunc(i)
                    return
                self.started = False
                self.forcePlayMusic("pacman_death.wav")
                reset()
            elif self.touchingPacman(ghost.row, ghost.col) and ghost.attacked and not ghost.dead:
                ghost.dead = True
                ghost.setTarget()
                ghost.ghostSpeed = 1
                ghost.row = math.floor(ghost.row)
                ghost.col = math.floor(ghost.col)
                self.score += self.ghostScore
                self.points.append([ghost.row, ghost.col, self.ghostScore, 0])
                self.ghostScore *= 2
                self.forcePlayMusic("eat_ghost.wav")
                time.sleep(0.4)
        if self.touchingPacman(self.berryLocation[0], self.berryLocation[1]) and \
                not self.berryState[2] and self.levelTimer in range(self.berryState[0], self.berryState[1]):
            self.berryState[2] = True
            self.score += self.berryScore
            self.points.append([self.berryLocation[0], self.berryLocation[1], self.berryScore, 0])
            self.berriesCollected.append(self.berries[(self.level - 1) % 8])
            self.forcePlayMusic("eat_fruit.wav")

    # Displays the current score
    def displayScore(self):
        scoreStart = 5
        highScoreStart = 11
        for i, char in enumerate("1UP"):
            tileImage = charsImages[char]
            screen.blit(tileImage, ((i + scoreStart) * square, 4, square, square))
        score = str(self.score)
        if score == "0":
            score = "00"
        for i, digit in enumerate(score):
            tileImage = charsImages[digit]
            screen.blit(tileImage, ((scoreStart + 2 + i) * square, square + 4, square, square))

        for i, char in enumerate("HIGH_SCORE"):
            tileImage = charsImages[char]
            screen.blit(tileImage, ((i + highScoreStart) * square, 4, square, square))

        highScore = str(self.highScore)
        if highScore == "0":
            highScore = "00"
        for i, digit in enumerate(highScore):
            tileImage = charsImages[digit]
            screen.blit(tileImage, ((highScoreStart + 6 + i) * square, square + 4, square, square))

    def drawBerry(self):
        if self.levelTimer in range(self.berryState[0], self.berryState[1]) and not self.berryState[2]:
            berryImage = loadElement(self.berries[(self.level - 1) % 8])
            berryImage = pygame.transform.scale(berryImage, (int(square * spriteRatio), int(square * spriteRatio)))
            screen.blit(berryImage, (self.berryLocation[1] * square, self.berryLocation[0] * square, square, square))

    @staticmethod
    def drawPoints(points, row, col):
        for i, digit in enumerate(str(points)):
            digit = "B" + digit
            tileImage = charsImages[digit]
            tileImage = pygame.transform.scale(tileImage, (square // 2, square // 2))
            screen.blit(tileImage,
                        (col * square + (square // 2 * i), row * square - 20, square // 2, square // 2))

    def gameOverFunc(self, repeat):
        global running
        if repeat == 12:
            running = False
            self.recordHighScore()
            return

        self.render()

        # Draws new image
        pacmanImage = loadElement(116 + repeat)
        pacmanImage = pygame.transform.scale(pacmanImage, (int(square * spriteRatio), int(square * spriteRatio)))
        screen.blit(pacmanImage,
                    (self.pacman.col * square + spriteOffset, self.pacman.row * square + spriteOffset, square, square))
        pygame.display.update()
        time.sleep(0.2)

    def displayLives(self):
        # 33 rows || 28 cols
        # Lives[[31, 5], [31, 3], [31, 1]]
        livesLoc = [[34, 3], [34, 1]]
        for i in range(self.lives - 1):
            lifeImage = loadElement(54)
            lifeImage = pygame.transform.scale(lifeImage, (int(square * spriteRatio), int(square * spriteRatio)))
            screen.blit(lifeImage, (livesLoc[i][1] * square, livesLoc[i][0] * square - spriteOffset, square, square))

    def displayBerries(self):
        firstBerrie = [34, 26]
        for i in range(len(self.berriesCollected)):
            berrieImage = loadElement(self.berriesCollected[i])
            berrieImage = pygame.transform.scale(berrieImage, (int(square * spriteRatio), int(square * spriteRatio)))
            screen.blit(berrieImage, ((firstBerrie[1] - (2 * i)) * square, firstBerrie[0] * square + 5, square, square))

    def touchingPacman(self, row, col):
        if row - 0.5 <= self.pacman.row <= row and col == self.pacman.col:
            return True
        elif row + 0.5 >= self.pacman.row >= row and col == self.pacman.col:
            return True
        elif row == self.pacman.row and col - 0.5 <= self.pacman.col <= col:
            return True
        elif row == self.pacman.row and col + 0.5 >= self.pacman.col >= col:
            return True
        elif row == self.pacman.row and col == self.pacman.col:
            return True
        return False

    def newLevel(self):
        reset()
        self.lives += 1
        self.collected = 0
        self.started = False
        self.berryState = [200, 400, False]
        self.levelTimer = 0
        self.lockedIn = True
        for level in self.levels:
            level[0] = min((level[0] + level[1]) - 100, level[0] + 50)
            level[1] = max(100, level[1] - 50)
        random.shuffle(self.levels)
        for i, state in enumerate(self.ghostStates):
            state[0] = randrange(2)
            state[1] = randrange(self.levels[i][state[0]] + 1)
        global gameBoard
        gameBoard = copy.deepcopy(originalGameBoard)
        self.render()

    # Flips Color of Special Tic-Taks
    @staticmethod
    def flipColor():
        global gameBoard
        for i in range(3, len(gameBoard) - 2):
            for j in range(len(gameBoard[0])):
                if gameBoard[i][j] == 5:
                    gameBoard[i][j] = 6
                    pygame.draw.circle(screen, pelletColor, (j * square + square // 2, i * square + square // 2),
                                       square // 2)
                elif gameBoard[i][j] == 6:
                    gameBoard[i][j] = 5
                    pygame.draw.circle(screen, (0, 0, 0), (j * square + square // 2, i * square + square // 2),
                                       square // 2)

    @staticmethod
    def getCount():
        total = 0
        for i in range(3, len(gameBoard) - 2):
            for j in range(len(gameBoard[0])):
                if gameBoard[i][j] == 2 or gameBoard[i][j] == 5 or gameBoard[i][j] == 6:
                    total += 1
        return total

    @staticmethod
    def getHighScore():
        file = open(DataPath + "HighScore.txt", "r")
        highScore = int(file.read())
        file.close()
        return highScore

    def recordHighScore(self):
        file = open(DataPath + "HighScore.txt", "w+")
        file.write(str(self.highScore))
        file.close()

    def displayFingers(self):
        pygame.draw.circle(screen, (255, 0, 0), self.fingers[0], square // 2)
        pygame.draw.circle(screen, (255, 0, 0), self.fingers[1], square // 2)

    def getTarget(self):
        ret, frame = cap.read()
        imgBGR = np.fliplr(frame)
        # переводим его в формат RGB для распознавания
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        m, n = imgRGB.shape[:2]
        # Распознаем
        results = handsDetector.process(imgRGB)
        imgRGB.fill(0)
        indexTip = (0, 0)
        thumbTip = (n, m)
        tap = False
        ignore = 0.2
        if results.multi_hand_landmarks is not None:
            indexTip = results.multi_hand_landmarks[0].landmark[8]
            indexTip = (indexTip.x - ignore) / (1 - 2 * ignore), (indexTip.y - ignore) / (1 - 2 * ignore)
            thumbTip = results.multi_hand_landmarks[0].landmark[4]
            thumbTip = (thumbTip.x - ignore) / (1 - 2 * ignore), (thumbTip.y - ignore) / (1 - 2 * ignore)
            dis = (thumbTip[0] * n - indexTip[0] * n) ** 2 + (thumbTip[1] * m - indexTip[1] * m) ** 2
            if dis < 3000:
                tapPoint = (thumbTip[0] + indexTip[0]) / 2, (thumbTip[1] + indexTip[1]) / 2
                self.pacman.setTarget(int(tapPoint[1] * height) // square, int(tapPoint[0] * width) // square)
                tap = True
        indexTip = (indexTip[0] * width, indexTip[1] * height)
        thumbTip = (thumbTip[0] * width, thumbTip[1] * height)
        self.fingers = [indexTip, thumbTip]
        return tap


class Pacman:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.mouthOpen = False
        self.pacSpeed = 1 / 4
        self.mouthChangeDelay = 5
        self.mouthChangeCount = 0
        self.dir = 0  # 0: North, 1: East, 2: South, 3: West
        self.target = (-1, -1)

    @staticmethod
    def isValid(cRow, cCol):
        if cCol < 0 or cCol > len(gameBoard[0]) - 1:
            return True
        for ghost in game.ghosts:
            if int(ghost.row) == cRow and int(ghost.col) == cCol and not ghost.attacked and not ghost.dead:
                return False
        if not ghostGate.count([cRow, cCol]) == 0:
            return False
        if gameBoard[cRow][cCol] == 3:
            return False
        return True

    def setTarget(self, row, col):
        row = max(0, row)
        row = min(len(gameBoard) - 1, row)
        col = max(0, col)
        col = min(len(gameBoard[0]) - 1, col)
        if self.isValid(row, col):
            self.target = (row, col)

    def setDir(self):
        targetDistances = [[INF] * len(gameBoard[0]) for _ in range(len(gameBoard))]
        targetDistances[self.target[0]][self.target[1]] = 0
        furthestPoints = deque()
        furthestPoints.append(self.target)

        while len(furthestPoints):
            point = furthestPoints[0]
            furthestPoints.popleft()
            dis = targetDistances[point[0]][point[1]]
            for newDir in range(4):
                newPoint = (point[0] + movements[newDir][0], (point[1] + movements[newDir][1]) % len(gameBoard[0]))
                isCoin = gameBoard[newPoint[0]][newPoint[1]] in [2, 5, 6]
                if self.isValid(*newPoint) and targetDistances[newPoint[0]][newPoint[1]] == INF:
                    targetDistances[newPoint[0]][newPoint[1]] = dis + (not isCoin)
                    if isCoin:
                        furthestPoints.appendleft(newPoint)
                    else:
                        furthestPoints.append(newPoint)

        dirs = [[0, -self.pacSpeed, 0],
                [1, 0, self.pacSpeed],
                [2, self.pacSpeed, 0],
                [3, 0, -self.pacSpeed]
                ]

        random.shuffle(dirs)
        best = INF
        bestDir = -1
        for newDir in dirs:
            newPoint = self.row + newDir[1], self.col + newDir[2]
            if newDir[0] == 0 and self.col % 1.0 == 0:
                newPoint = math.floor(newPoint[0]), int(newPoint[1])
            elif newDir[0] == 1 and self.row % 1.0 == 0:
                newPoint = int(newPoint[0]), math.ceil(newPoint[1])
            elif newDir[0] == 2 and self.col % 1.0 == 0:
                newPoint = math.ceil(newPoint[0]), int(newPoint[1])
            elif newDir[0] == 3 and self.row % 1.0 == 0:
                newPoint = int(newPoint[0]), math.floor(newPoint[1])
            else:
                continue

            newPoint = newPoint[0], newPoint[1] % len(gameBoard[0])
            dis = targetDistances[newPoint[0]][newPoint[1]]
            if dis < best and self.isValid(newPoint[0], newPoint[1]):
                bestDir = newDir[0]
                best = dis

        self.dir = bestDir

    def update(self):
        if self.target == (-1, -1) or (self.row == self.target[0] and self.col == self.target[1]):
            return
        self.setDir()
        if self.dir != -1:
            self.row += self.pacSpeed * movements[self.dir][0]
            self.col += self.pacSpeed * movements[self.dir][1]
            self.col = self.col % len(gameBoard[0])

    # Draws pacman based on his current state
    def draw(self):
        if not game.started:
            pacmanImage = loadElement(112)
            pacmanImage = pygame.transform.scale(pacmanImage, (int(square * spriteRatio), int(square * spriteRatio)))
            screen.blit(pacmanImage,
                        (self.col * square + spriteOffset, self.row * square + spriteOffset, square, square))
            return

        if self.mouthChangeCount == self.mouthChangeDelay:
            self.mouthChangeCount = 0
            self.mouthOpen = not self.mouthOpen
        self.mouthChangeCount += 1
        if self.mouthOpen:
            pacmanImage = loadElement(49)
        else:
            pacmanImage = loadElement(51)
        for i in range(self.dir):
            pacmanImage = pygame.transform.rotate(pacmanImage, -90)

        pacmanImage = pygame.transform.scale(pacmanImage, (int(square * spriteRatio), int(square * spriteRatio)))
        screen.blit(pacmanImage, (self.col * square + spriteOffset, self.row * square + spriteOffset, square, square))
        if self.target != (-1, -1):
            pygame.draw.circle(screen, (0, 255, 0),
                               ((self.target[1] + 0.5) * square, (self.target[0] + 0.5) * square), square // 2)


class Ghost:
    def __init__(self, row, col, color, changeFeetCount):
        self.row = row
        self.col = col
        self.attacked = False
        self.color = color
        self.dir = randrange(4)
        self.dead = False
        self.changeFeetCount = changeFeetCount
        self.changeFeetDelay = 5
        self.target = (-1, -1)
        self.ghostSpeed = 1 / 4
        self.lastLoc = (-1, -1)
        self.attackedTimer = 240
        self.attackedCount = 0
        self.deathTimer = 120
        self.deathCount = 0

    def update(self):
        if self.target == (-1, -1) or (self.row == self.target[0] and self.col == self.target[1]) or \
                gameBoard[int(self.row)][int(self.col)] == 4 or self.dead:
            self.setTarget()
        self.setDir()
        self.move()

        if self.attacked:
            self.attackedCount += 1

        if self.attacked and not self.dead:
            self.ghostSpeed = 1 / 8

        if self.attackedCount == self.attackedTimer and self.attacked:
            if not self.dead:
                self.ghostSpeed = 1 / 4
                self.row = math.floor(self.row)
                self.col = math.floor(self.col)

            self.attackedCount = 0
            self.attacked = False
            self.setTarget()

        if self.dead and gameBoard[self.row][self.col] == 4:
            self.deathCount += 1
            self.attacked = False
            if self.deathCount == self.deathTimer:
                self.deathCount = 0
                self.dead = False
                self.ghostSpeed = 1 / 4

    def draw(self, moving):  # Ghosts states: Alive, Attacked, Dead Attributes: Color, Direction, Location
        currentDir = ((self.dir + 3) % 4) * 2
        if self.changeFeetCount == self.changeFeetDelay:
            self.changeFeetCount = 0
            currentDir += 1
        self.changeFeetCount += moving
        if self.dead:
            tileNum = 152 + currentDir
            ghostImage = loadElement(tileNum)
        elif self.attacked:
            if self.attackedTimer - self.attackedCount < self.attackedTimer // 3 and \
                    (self.attackedTimer - self.attackedCount) % 31 < 26:
                ghostImage = loadElement(70 + currentDir - (((self.dir + 3) % 4) * 2))
            else:
                ghostImage = loadElement(72 + currentDir - (((self.dir + 3) % 4) * 2))
        else:
            if self.color == "blue":
                tileNum = 136 + currentDir
            elif self.color == "pink":
                tileNum = 128 + currentDir
            elif self.color == "orange":
                tileNum = 144 + currentDir
            else:
                tileNum = 96 + currentDir
            ghostImage = loadElement(tileNum)

        ghostImage = pygame.transform.scale(ghostImage, (int(square * spriteRatio), int(square * spriteRatio)))
        screen.blit(ghostImage, (self.col * square + spriteOffset, self.row * square + spriteOffset, square, square))

    def isValid(self, cRow, cCol):
        if cCol < 0 or cCol > len(gameBoard[0]) - 1:
            return True
        for ghost in game.ghosts:
            if ghost.color == self.color:
                continue
            if int(ghost.row) == cRow and int(ghost.col) == cCol and not self.dead:
                return False
        if not ghostGate.count([cRow, cCol]) == 0:
            if self.dead and self.row < cRow:
                return True
            elif self.row > cRow and not self.dead and not self.attacked and not game.lockedIn:
                return True
            else:
                return False
        if gameBoard[cRow][cCol] == 3:
            return False
        return True

    def setDir(self):  # Not best route but a route nonetheless
        dirs = [[0, -self.ghostSpeed, 0],
                [1, 0, self.ghostSpeed],
                [2, self.ghostSpeed, 0],
                [3, 0, -self.ghostSpeed]
                ]
        random.shuffle(dirs)
        best = INF
        bestDir = -1
        for newDir in dirs:
            newPoint = self.row + newDir[1], self.col + newDir[2]
            if newDir[0] == 0 and self.col % 1.0 == 0:
                newPoint = math.floor(newPoint[0]), int(newPoint[1])
            elif newDir[0] == 1 and self.row % 1.0 == 0:
                newPoint = int(newPoint[0]), math.ceil(newPoint[1])
            elif newDir[0] == 2 and self.col % 1.0 == 0:
                newPoint = math.ceil(newPoint[0]), int(newPoint[1])
            elif newDir[0] == 3 and self.row % 1.0 == 0:
                newPoint = int(newPoint[0]), math.floor(newPoint[1])
            else:
                continue

            dis = calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
            if dis < best and self.isValid(newPoint[0], newPoint[1]) and \
                    not (self.lastLoc[0] == self.row + newDir[1] and self.lastLoc[1] == self.col + newDir[2]):
                bestDir = newDir[0]
                best = dis

        self.dir = bestDir

    def setTarget(self):
        if gameBoard[int(self.row)][int(self.col)] == 4 and not self.dead:
            self.target = (ghostGate[0][0] - 1, ghostGate[0][1] + 1)
            return
        elif gameBoard[int(self.row)][int(self.col)] == 4 and self.dead:
            self.target = (self.row, self.col)
        elif self.dead:
            self.target = (14, 13)
            return

        # Records the quadrants of each ghost's target
        quads = [0, 0, 0, 0]
        for ghost in game.ghosts:
            if ghost.target[0] <= 15 and ghost.target[1] >= 13:
                quads[0] += 1
            elif ghost.target[0] <= 15 and ghost.target[1] < 13:
                quads[1] += 1
            elif ghost.target[0] > 15 and ghost.target[1] < 13:
                quads[2] += 1
            elif ghost.target[0] > 15 and ghost.target[1] >= 13:
                quads[3] += 1

        # Finds a target that will keep the ghosts dispersed
        while True:
            self.target = (randrange(31), randrange(28))
            quad = 0
            if self.target[0] <= 15 and self.target[1] >= 13:
                quad = 0
            elif self.target[0] <= 15 and self.target[1] < 13:
                quad = 1
            elif self.target[0] > 15 and self.target[1] < 13:
                quad = 2
            elif self.target[0] > 15 and self.target[1] >= 13:
                quad = 3
            if not gameBoard[self.target[0]][self.target[1]] == 3 and \
                    not gameBoard[self.target[0]][self.target[1]] == 4:
                break
            elif quads[quad] == 0:
                break

    def move(self):
        self.lastLoc = (self.row, self.col)
        if self.dir != -1:
            self.row += self.ghostSpeed * movements[self.dir][0]
            self.col += self.ghostSpeed * movements[self.dir][1]
            self.col = self.col % len(gameBoard[0])


game = Game(1, 0)
ghostSafeArea = [15, 13]  # The location the ghosts escape to when attacked
ghostGate = [[15, 13], [15, 14]]


def canMove(row, col):
    if col == -1 or col == len(gameBoard[0]):
        return True
    if gameBoard[int(row)][int(col)] != 3:
        return True
    return False


# Reset after death
def reset():
    global game
    game.ghosts = [Ghost(14.0, 13.5, "red", 0), Ghost(17.0, 11.5, "blue", 1),
                   Ghost(17.0, 13.5, "pink", 2), Ghost(17.0, 15.5, "orange", 3)]
    for ghost in game.ghosts:
        ghost.setTarget()
    game.pacman = Pacman(26.0, 13.5)
    game.lives -= 1
    game.paused = True
    game.render()


def displayLaunchScreen():
    launchImage = pygame.image.load(SheetsPath + "Launch.png")
    launchImage = pygame.transform.scale(launchImage, (width, height))
    screen.blit(launchImage, (0, 0))
    pygame.display.update()


running = True
onLaunchScreen = True
displayLaunchScreen()
clock = pygame.time.Clock()

frames = 0

while running:
    frames += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False
            game.recordHighScore()
        elif event.type == pygame.KEYDOWN:
            if onLaunchScreen and event.key == pygame.K_SPACE:
                onLaunchScreen = False
                game.paused = True
                game.started = False
                pygame.mixer.music.load(MusicPath + "pacman_beginning.wav")
                pygame.mixer.music.play()

    if not onLaunchScreen:
        game.update()
    clock.tick(FPS)
