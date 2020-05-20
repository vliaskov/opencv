#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import video
from common import anorm2, draw_str

import pygame
from pygame import gfxdraw
import sys
import math
import random
import numpy
import time
from vector_2d import Vector

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Motion detection
# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10
# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 1000
MAX_SIZE_FOR_MOVEMENT = 20000
# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 100

class Particle(object):
    """Paticle like Object"""

    def __init__(self, pos, size, color, direction, width, height, gravity, surface, screen, drag=0.999, elasticity=0.75, density=1):
        #self.surface = surface
        self.pos = pos
        self.size = size
        self.color=color
        self.direction = direction
        self.gravity = gravity
        self.drag = drag
        self.elasticity = elasticity
        # density and mass
        self.density = 1
        self.mass = self.density * self.size ** 2
        # set color according to mass
        self.color = pygame.Color(200 - density * 10, 200 - density * 10, 255)
        self.width = width
        self.height = height
        #self.speed = speed
        #self.angle = angle
        # initialize things
        self.surface = surface
        self.screen = screen
        self.font = pygame.font.SysFont(None, 24)
        self.text = self.font.render('hello', True, (100, 255, 250))



    def bounce(self):
        width = self.width
        height = self.height
        # right border
        if self.pos.x > width - self.size:
            self.pos += Vector(2*(width - self.size) - self.pos.x, self.pos.y)
            #self.direction += Vector(-2 * self.direction.x, self.direction.y)
            #self.direction *= self.elasticity
        # left border
        elif self.pos.x < self.size:
            self.pos += Vector(2*self.size - self.pos.x, self.pos.y)
            #self.direction += Vector(-2 * self.direction.x, self.direction.y)
            #self.direction.bounce(0)
            #self.direction *= self.elasticity
        # lower border
        if self.pos.y > height - self.size:
            self.pos += Vector(self.pos.x, 2*(height - self.size) - self.pos.y)
            # TODO why
            #self.direction.bounce(math.pi)
            self.direction += Vector(self.direction.x, -2 * self.direction.y)
            #self.direction.angle = math.pi - self.direction.angle
            #self.direction *= self.elasticity
        # upper border
        elif self.pos.y < self.size:
            self.pos = Vector(self.pos.x, 2*self.size - self.pos.y)
            # TODO why
            #self.direction += Vector(self.direction.x, -2 * self.direction.y)
            #self.direction.bounce(math.pi)
            #self.direction.angle = math.pi - self.direction.angle
            #self.direction *= self.elasticity

    #def check_interactions(self, point):
        

    def move(self):
        #self.pos = Vector(self.pos.x + math.sin(self.angle) * self.speed, self.pos.y + math.cos(self.angle) * self.speed)
        # add gravity
        self.pos += self.direction
        self.pos += self.gravity
        # add drag
        #self.direction *= self.drag

    def update(self, image):#**kwds):
        self.move()
        self.bounce()
        #dirtyrect = pygame.draw.circle(self.surface, self.color, Vec2d(int(self.pos.x), int(self.pos.y)), self.size, 1)
        #cv.circle(image, (self.pos.x, self.pos.y), self.color, 1, 8, 0) #self.size, 1)
        cv.circle(image, (int(self.pos.x), int(self.pos.y)), 30, (0, 255, 255), 1, 8, 0) #self.size, 1)
        #screen.blit(self.text, self.pos))

    def render_text(self):
        self.surface.blit(self.text, pygame.Rect((self.pos.x - self.size/2, self.pos.y - self.size / 2), (self.pos.x + self.size / 2, self.pos.y + self.size / 2)))
        self.screen.blit(self.surface, (0,0))

    def __repr__(self):
        return("Particle(%(surface)s, %(pos)s, %(size)s, %(color)s)" % self.__dict__)

class App:
    def __init__(self, video_src, screen, surface):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.width  = self.cam.get(cv.CAP_PROP_FRAME_WIDTH)  # float
        self.height = self.cam.get(cv.CAP_PROP_FRAME_HEIGHT) # float
        self.frame_idx = 0
        self.game_state = 0
        self.game_time = 0
        o1 = Vector(120, 100)
        o2 = Vector(500,400) 
        self.objects = [
            #0: [Vector(120, 100), Vector(500,400)] 
            [o1, o2]  #0
	]
        self.particles = []
        self.count = 4
        self.elasticity = 0.75
        self.drag = 0.999
        self.gravity = Vector(0.001, math.pi)
        self.screen = screen
        self.surface = surface
        self.font = pygame.font.SysFont(None, 24)
        self.text = self.font.render('hello', True, (100, 255, 250))

        
    def render_text_objects(self):
        for part in self.particles:
            part.render_text()

    def update_interactions(self, image):
        areas = self.objects[self.game_state]
        idx = 0
        for part in self.particles:
            part.update(image)
        #for obj in areas:
         #   directionx = random.randint(-10,10)# / 100;
            #directiony = random.randint(-10,10)# / 100;
            #areas.remove(obj)
            #obj = Vector(obj.x + directionx, obj.y + directiony)
            #areas.append(obj)

    def set_interactions(self):
        areas = self.objects[self.game_state]
        return areas
        
    def check_interactions(self, areas, motioncenters, dist):
        activated = []
        idle = areas
        for area in areas:
            for c in motioncenters:
                #if np.sqrt(np.sum((area - c) ** 2)) < dist:
                if (abs(area.x - c.x) < dist) and (abs(area.y - c.y) < dist):
                    activated.append(area)
                    idle.remove(area)
        return activated, idle
                 
    def draw_idle_interactions(self, image, idleareas, radius):
        for area in idleareas:
            cv.circle(image, (area.x, area.y), radius, (255, 255, 0), 1, 8, 0)

    def run(self):
	# Init frame variables for motion detection
        first_frame = None
        next_frame = None
        delay_counter = 0
        movement_persistent_counter = 0

        #moving objects
        for i in range(self.count):
            pos = Vector(random.randint(0, self.width), random.randint(0, self.height))
            color = pygame.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
            size = 10 + random.randint(0, 20)
            speed = random.randint(0, 4)
            angle = random.uniform(0, math.pi * 2)
            density = random.randint(0, 20)
            self.particles.append(Particle(pos, size, color, Vector(speed, angle), self.width, self.height, self.gravity, self.surface, self.screen, self.drag, self.elasticity, density))
        
        #immovable objects
        for i in range(self.count):
            pos = Vector(random.randint(0, self.width), random.randint(0, self.height))
            color = pygame.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
            size = 10 + random.randint(0, 20)
            speed = 0
            angle = 0
            density = random.randint(0, 20)
            self.particles.append(Particle(pos, size, color, Vector(speed, angle), self.width, self.height, Vector(0, 0), self.surface, self.screen, self.drag, self.elasticity, density))

        circleradius = 50;
        interactions = self.set_interactions()
        idle = interactions
	
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            dark = frame.copy()
            dark[:,:] = (0, 0, 0)

            # Set transient motion detected as false
            transient_movement_flag = False

            gray = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)

    	    # If the first frame is nothing, initialise it
            if first_frame is None: first_frame = gray
            delay_counter += 1
	    # Otherwise, set the first frame to compare as the previous frame
	    # But only if the counter reaches the appriopriate value
	    # The delay is to allow relatively slow motions to be counted as large
	    # motions if they're spread out far enough
            if delay_counter > FRAMES_TO_PERSIST:
                delay_counter = 0
                first_frame = next_frame
	    # Set the next frame to compare (the current frame)
            next_frame = gray
	    # Compare the two frames, find the difference
            frame_delta = cv.absdiff(first_frame, next_frame)
            thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]
            # Fill in holes via dilate(), and find contours of the thesholds
            thresh = cv.dilate(thresh, None, iterations = 2)
            cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for c in cnts:
                # Save the coordinates of all found contours
                (x, y, w, h) = cv.boundingRect(c)
                # If the contour is too small, ignore it, otherwise, there's transient
                # movement
                if cv.contourArea(c) > MIN_SIZE_FOR_MOVEMENT: # and cv.contourArea(c) < MAX_SIZE_FOR_MOVEMENT:
                    #if cv.contourArea(c) > MAX_SIZE_FOR_MOVEMENT:
                    #    print("Large Contour found")
                    transient_movement_flag = True
                    # Draw a rectangle around big enough movements
                    #cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cvlist = [c]
                    cv.drawContours(vis, cvlist, 0, (0, 0, 255), 3 )
                    cv.drawContours(dark, cvlist, 0, (0, 0, 255), 3 )
                    activated, idle = self.check_interactions(interactions, [Vector(x + int(w/2), y + int(h/2))], circleradius)

	    # The moment something moves momentarily, reset the persistent
	    # movement timer.
            if transient_movement_flag == True:
                movement_persistent_flag = True
                movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

	    # The canny filter is for edge detection
            thrs1 = 2500;#cv.getTrackbarPos('thrs1', 'edge')
            thrs2 = 4500;#cv.getTrackbarPos('thrs2', 'edge')
            edge = cv.Canny(gray, thrs1, thrs2, apertureSize=5)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []


                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    cv.circle(dark, (x, y), 2, (0, 255, 0), -1)
                    #activated, idle = self.check_interactions(interactions, [Vector(x, y)], circleradius)

                self.tracks = new_tracks

                vis = np.uint8(vis/2.)
                vis[edge != 0] = (255, 0, 0)

                dark[edge != 0] = (255, 0, 0)
                #black out video except edges and tracked points 
                #vis[edge == 0] = (0, 0, 0)
                #vis[edge == 0] = vis[edge == 0] / 5

                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                cv.polylines(dark, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))


                #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
		
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.update_interactions(vis)
            self.draw_idle_interactions(vis, idle, circleradius)
            self.update_interactions(dark)
            self.draw_idle_interactions(dark, idle, circleradius)

            self.frame_idx += 1
            self.prev_gray = frame_gray
            #cv.imshow('lk_track', vis)
            #cv.imshow('lk_track', dark)
            cv.imwrite("frame.jpg", dark)
            #darkimg = pygame.image.frombuffer(dark, (self.width, self.height), "RGB")
            darkimg = pygame.image.load("frame.jpg")
            self.surface.blit(darkimg, pygame.Rect((0,0), (self.screen.get_size()[0], self.screen.get_size()[1]) ))
            self.screen.blit(self.surface, (0,0))

            doorimg = pygame.image.load("castledoors.png") #cv.imread("door.jpeg")
            doorimg = doorimg.convert()
            doorRect = pygame.Rect((10,10),(20, 20))
            self.surface.blit(doorimg, doorRect)
            self.screen.blit(self.surface, (0,0))

            self.render_text_objects()
            #self.surface.blit(self.text, pygame.Rect((600,400), (650, 500)))
            #self.screen.blit(self.surface, (0,0))

            #pygame.display.flip()
            pygame.display.update()

            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    pygame.init()
    screen = pygame.display.set_mode((720, 480))
    pygame.display.set_caption('My game')

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))
    #pygame.display.flip()

    App(video_src, screen, background).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
