# -*- coding: utf-8 -*-

import math
import random
import sys
import time

import numpy as np
import pygame


class LeakParticle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    def __init__(self, x, y, vx, vy, life, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color


class Graphics:
    def __init__(self,device_connected,window_size=(600,400)):
        self.device_connected = device_connected
        
        #initialize pygame window
        self.window_size = window_size #default (600,400)
        pygame.init()
        self.window = pygame.display.set_mode((window_size[0]*2, window_size[1]))   ##twice 600x400 for haptic and VR
        pygame.display.set_caption('Virtual Haptic Device')

        self.screenHaptics = pygame.Surface(self.window_size)
        self.screenVR = pygame.Surface(self.window_size)

        ##add nice icon from https://www.flaticon.com/authors/vectors-market
        self.icon = pygame.image.load('robot.png')
        pygame.display.set_icon(self.icon)

        ##add text on top to debugToggle the timing and forces
        self.font = pygame.font.Font('freesansbold.ttf', 18)

        pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' toggles it
         
        ##set up the on-screen debugToggle
        self.text = self.font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (10, 10)

        #xc,yc = screenVR.get_rect().center ##center of the screen

        ##initialize "real-time" clock
        self.clock = pygame.time.Clock()
        self.FPS = 100   #in Hertz

        ##define some colors
        self.cWhite = (255,255,255)
        self.cDarkblue = (36,90,190)
        self.cLightblue = (0,176,240)
        self.cRed = (255,0,0)
        self.cOrange = (255,100,0)
        self.cYellow = (255,255,0)
        
        self.hhandle = pygame.image.load('handle.png') #
        
        self.haptic_width = 48
        self.haptic_height = 48
        self.haptic  = pygame.Rect(*self.screenHaptics.get_rect().center, 0, 0).inflate(self.haptic_width, self.haptic_height)
        self.effort_cursor  = pygame.Rect(*self.haptic.center, 0, 0).inflate(self.haptic_width, self.haptic_height) 
        self.colorHaptic = self.cOrange ##color of the wall
        self.effort_color = (255, 255, 255)

        ####Pseudo-haptics dynamic parameters, k/b needs to be <1
        self.sim_k = 0.5 #0.1#0.5       ##Stiffness between cursor and haptic display
        self.sim_b = 0.8 #1.5#0.8       ##Viscous of the pseudohaptic display
        
        self.window_scale = 5000 #pixels per meter
        self.device_origin = (int(self.window_size[0]/2.0 + 0.038/2.0*self.window_scale),0)
        
        self.show_linkages = True
        self.show_debug = True
        self.hide_haptic_vr = False  # set True to suppress haptic square on screenVR

        # ── Mars greenhouse VR panel ────────────────────────────────────
        self.anim_time = 0.0
        # The breach is on the pressurized greenhouse envelope. The top of the
        # scene shows the Martian exterior; the lower scene shows the glazing,
        # structure and crop bay inside the greenhouse.
        self.MARS_WALL = (124, 146, 142)
        self.MARS_WALL_LT = (188, 210, 204)
        self.MARS_WALL_DK = (56, 78, 74)
        self.MARS_DUST = (178, 168, 154)
        self.VIEWPORT_METAL = (104, 118, 112)
        self.VIEWPORT_SHADOW = (32, 38, 36)
        self.HULL_TRIM = (156, 176, 170)
        self.HULL_STRIPE = (200, 168, 72)
        self.HULL_STRIPE_DK = (66, 58, 34)
        self.CRACK_GLOW = (255, 80, 30)
        self.CRACK_CORE = (255, 160, 60)
        self.CRACK_HOT = (255, 228, 180)
        self.HUD_CYAN = (110, 205, 235)
        self.HUD_ORANGE = (255, 180, 40)
        self.STAR_COLOR = (255, 255, 255)
        self.WARNING_YELLOW = (222, 176, 64)
        self.WARNING_BLACK = (26, 26, 26)
        self.LEAK_COLOR = (210, 225, 255)
        self.LEAK_HOT = (255, 210, 160)
        self.HULL_ACCENT = (82, 190, 154)
        self.HULL_ACCENT_DK = (28, 84, 68)
        self.BULKHEAD_CORE = (72, 88, 82)
        self.BULKHEAD_INNER = (40, 52, 48)
        self.SCORCH = (52, 28, 22)

        self.stars = [
            (
                random.randint(0, window_size[0]),
                random.randint(0, window_size[1] // 3),
                random.uniform(0.5, 2.5),
                random.randint(150, 255),
                random.uniform(0.0, math.tau),
            )
            for _ in range(80)
        ]
        self.dust_motes = []
        self.leak_particles = []
        self.max_leak_particles = 180
        self._generate_mars_terrain()
        self._generate_wall_texture()
        self._generate_wall_panels()
        self._generate_dust_motes()
        self._generate_porthole()

    def convert_pos(self,*positions):
        #invert x because of screen axes
        # 0---> +X
        # |
        # |
        # v +Y
        converted_positions = []
        for physics_pos in positions:
            x = self.device_origin[0]-physics_pos[0]*self.window_scale
            y = self.device_origin[1]+physics_pos[1]*self.window_scale
            converted_positions.append([x,y])
        if len(converted_positions)<=0:
            return None
        elif len(converted_positions)==1:
            return converted_positions[0]
        else:
            return converted_positions
        return [x,y]
    def inv_convert_pos(self,*positions):
        #convert screen positions back into physical positions
        converted_positions = []
        for screen_pos in positions:
            x = (self.device_origin[0]-screen_pos[0])/self.window_scale
            y = (screen_pos[1]-self.device_origin[1])/self.window_scale
            converted_positions.append([x,y])
        if len(converted_positions)<=0:
            return None
        elif len(converted_positions)==1:
            return converted_positions[0]
        else:
            return converted_positions
        return [x,y]
        
    def get_events(self):
        #########Process events  (Mouse, Keyboard etc...)#########
        events = pygame.event.get()
        keyups = []
        for event in events:
            if event.type == pygame.QUIT: #close window button was pressed
                sys.exit(0) #raises a system exit exception so any Finally will actually execute
            elif event.type == pygame.KEYUP:
                keyups.append(event.key)
        
        mouse_pos = pygame.mouse.get_pos()
        return keyups, mouse_pos

    def sim_forces(self,pE,f,pM,mouse_k=None,mouse_b=None):
        #simulated device calculations
        if mouse_k is not None:
            self.sim_k = mouse_k
        if mouse_b is not None:
            self.sim_b = mouse_b
        if not self.device_connected:
            pP = self.haptic.center
            #pM is where the mouse is
            #pE is where the position is pulled towards with the spring and damping factors
            #pP is where the actual haptic position ends up as
            diff = np.array(( pM[0]-pE[0],pM[1]-pE[1]) )
            #diff = np.array(( pM[0]-pP[0],pM[1]-pP[1]) )
            
            scale = self.window_scale/1e3
            scaled_vel_from_force = np.array(f)*scale/self.sim_b
            vel_from_mouse_spring = (self.sim_k/self.sim_b)*diff
            dpE = vel_from_mouse_spring - scaled_vel_from_force
            #dpE = -dpE
            #if diff[0]!=0:
            #    if (diff[0]+dpE[0])/diff[0]<0:
            #        #adding dpE has changed the sign (meaning the distance that will be moved is greater than the original displacement
            #        #prevent the instantaneous velocity from exceeding the original displacement (doesn't make physical sense)
            #        #basically if the force given is so high that in a single "tick" it would cause the endpoint to move back past it's original position...
            #        #whatever thing is exerting the force should basically be considered a rigid object
            #        dpE[0] = -diff[0]
            #if diff[1]!=1:
            #    if (diff[1]+dpE[1])/diff[1]<0:
            #        dpE[1] = -diff[1]
            if abs(dpE[0])<1:
                dpE[0] = 0
            if abs(dpE[1])<1:
                dpE[1] = 0
            pE = np.round(pE+dpE) #update new positon of the end effector
            
            #Change color based on effort
            cg = 255-np.clip(np.linalg.norm(self.sim_k*diff/self.window_scale)*255*20,0,255)
            cb = 255-np.clip(np.linalg.norm(self.sim_k*diff/self.window_scale)*255*20,0,255)
            self.effort_color = (255,cg,cb)
        return pE

    def _generate_mars_terrain(self):
        """Generate a simple Mars horizon silhouette for the VR panel."""
        w = self.window_size[0]
        base_y = self.window_size[1] // 3
        self.terrain_points = []
        for x in range(0, w + 20, 20):
            y = base_y + int(
                15 * math.sin(x * 0.008)
                + 8 * math.sin(x * 0.023)
                + 5 * math.sin(x * 0.05)
            )
            self.terrain_points.append((x, y))
        self.terrain_points.append((w, self.window_size[1]))
        self.terrain_points.append((0, self.window_size[1]))

    def _generate_wall_texture(self):
        """Pre-generate dust, condensation and wear for greenhouse glazing."""
        w, h = self.window_size
        self.wall_spots = []
        for _ in range(200):
            x = random.randint(0, w)
            y = random.randint(0, h)
            r = random.randint(2, 8)
            shade = random.randint(-15, 15)
            self.wall_spots.append((x, y, r, shade))

        self.wall_streaks = []
        for _ in range(55):
            x = random.randint(15, w - 15)
            y = random.randint(h // 3 + 30, h - 40)
            length = random.randint(25, 110)
            alpha = random.randint(12, 34)
            self.wall_streaks.append((x, y, length, alpha))

    def _generate_wall_panels(self):
        """Generate glazing segments, trusses and service hardware for the greenhouse shell."""
        w, h = self.window_size
        viewport_h = h // 3
        self.wall_panels = []
        self.wall_rivets = []
        self.warning_plates = []
        panel_w = 148
        panel_h = 116

        for py in range(viewport_h + 18, h - 14, panel_h):
            for px in range(12, w - 12, panel_w):
                jitter_x = random.randint(-8, 8)
                jitter_y = random.randint(-6, 6)
                rect = pygame.Rect(px + jitter_x, py + jitter_y, panel_w - 14, panel_h - 12)
                self.wall_panels.append(rect)
                corners = [
                    (rect.left + 10, rect.top + 10),
                    (rect.right - 10, rect.top + 10),
                    (rect.left + 10, rect.bottom - 10),
                    (rect.right - 10, rect.bottom - 10),
                ]
                self.wall_rivets.extend(corners)
                if random.random() < 0.2:
                    self.warning_plates.append(
                        pygame.Rect(rect.left + 12, rect.top + 12, 78, 18)
                    )

        self.vent_slots = []
        for _ in range(7):
            vx = random.randint(40, w - 150)
            vy = random.randint(viewport_h + 65, h - 90)
            self.vent_slots.append(pygame.Rect(vx, vy, random.randint(78, 120), 16))

        self.crop_beds = []
        bed_y = h - 108
        for px in range(34, w - 74, 104):
            self.crop_beds.append(pygame.Rect(px, bed_y + random.randint(-6, 6), 84, 34))

    def _generate_dust_motes(self):
        """Create slow moving dust inside the greenhouse volume."""
        w, h = self.window_size
        self.dust_motes = []
        for _ in range(70):
            self.dust_motes.append([
                random.uniform(0, w),
                random.uniform(h // 3 + 20, h - 30),
                random.uniform(0.6, 2.2),
                random.uniform(10.0, 45.0),
                random.uniform(-6.0, 6.0),
                random.uniform(0.25, 1.0),
            ])

    def _generate_porthole(self):
        """Pre-compute a side greenhouse inspection window with visible crop rows."""
        w, h = self.window_size
        viewport_h = h // 3
        self.porthole_cx = int(w * 0.14)
        self.porthole_cy = int(viewport_h + (h - viewport_h) * 0.50)
        self.porthole_r = min(int(w * 0.095), int((h - viewport_h) * 0.30))

        rng = random.Random(7)
        self.porthole_stars = [
            (rng.uniform(-0.90, 0.90), rng.uniform(-0.90, 0.90), rng.randint(1, 2), rng.randint(100, 170))
            for _ in range(18)
        ]
        self.window_leaf_clusters = []
        for _ in range(18):
            self.window_leaf_clusters.append(
                (
                    rng.uniform(-0.62, 0.62),
                    rng.uniform(-0.35, 0.72),
                    rng.randint(7, 16),
                    rng.choice([(78, 148, 82), (98, 172, 96), (132, 194, 112)]),
                )
            )

    def _draw_porthole(self, surface):
        """Draw a circular inspection window looking into an adjacent crop bay."""
        cx = self.porthole_cx
        cy = self.porthole_cy
        r = self.porthole_r
        d = r * 2

        port = pygame.Surface((d, d), pygame.SRCALPHA)
        port.fill((18, 26, 20))

        for y in range(d):
            t = y / max(1, d - 1)
            col = (
                int(32 + 14 * (1.0 - t)),
                int(46 + 26 * (1.0 - t)),
                int(28 + 12 * (1.0 - t)),
            )
            pygame.draw.line(port, col, (0, y), (d, y))

        for shelf_y in (int(d * 0.58), int(d * 0.76)):
            pygame.draw.line(port, (88, 98, 84), (14, shelf_y), (d - 14, shelf_y), 4)

        for rel_x, rel_y, leaf_r, color in self.window_leaf_clusters:
            px = int(r + rel_x * r)
            py = int(r + rel_y * r)
            pygame.draw.ellipse(port, color, (px - leaf_r, py - leaf_r // 2, leaf_r * 2, leaf_r))
            pygame.draw.line(port, (178, 196, 154), (px, py + leaf_r // 2), (px, py + leaf_r), 1)

        for sx_n, sy_n, ssize, sbright in self.porthole_stars:
            px = int(r + sx_n * r)
            py = int(r + sy_n * r)
            if 0 < px < d and 0 < py < d:
                tw = int(sbright * (0.82 + 0.18 * math.sin(self.anim_time * 1.1 + px * 0.07)))
                tw = max(0, min(255, tw))
                pygame.draw.circle(port, (tw, tw, tw), (px, py), ssize)

        clip = pygame.Surface((d, d), pygame.SRCALPHA)
        clip.fill((0, 0, 0, 0))
        pygame.draw.circle(clip, (255, 255, 255, 255), (r, r), r)
        port.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(port, (cx - r, cy - r))

        pygame.draw.circle(surface, (22, 26, 34), (cx, cy), r + 17)
        pygame.draw.circle(surface, (66, 82, 76), (cx, cy), r + 13)
        pygame.draw.circle(surface, (122, 138, 132), (cx, cy), r + 10)
        pygame.draw.circle(surface, (84, 100, 96), (cx, cy), r + 7)
        pygame.draw.circle(surface, (36, 42, 52), (cx, cy), r + 4)
        pygame.draw.circle(surface, (14, 16, 22), (cx, cy), r + 1, 2)

        for i in range(8):
            angle = math.pi / 8 + i * math.tau / 8
            bx = int(cx + (r + 11) * math.cos(angle))
            by = int(cy + (r + 11) * math.sin(angle))
            pygame.draw.circle(surface, (28, 32, 40), (bx, by), 5)
            pygame.draw.circle(surface, (126, 136, 148), (bx - 1, by - 1), 2)

        gloss = pygame.Surface((d, d), pygame.SRCALPHA)
        gloss_pts = [(r // 3, r // 5), (r, r // 7),
                     (r * 3 // 4, r // 2), (r // 5, r // 2)]
        pygame.draw.polygon(gloss, (255, 255, 255, 20), gloss_pts)
        gloss.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(gloss, (cx - r, cy - r))

        pf = pygame.font.SysFont("Consolas", 10, bold=True)
        lbl = pf.render("ADJ. GROWTH BAY", True, (82, 94, 110))
        surface.blit(lbl, (cx - lbl.get_width() // 2, cy + r + 18))

    def _draw_space_background(self, surface):
        """Draw a Mars sky viewport with stars and horizon."""
        w, h = self.window_size
        viewport_h = h // 3

        for y in range(viewport_h):
            t = y / max(viewport_h, 1)
            r = int(5 + t * 60)
            g = int(5 + t * 25)
            b = int(15 + t * 15)
            pygame.draw.line(surface, (r, g, b), (0, y), (w, y))

        # Distant planet glow near the horizon
        sun_x = int(w * 0.78)
        sun_y = int(viewport_h * 0.62)
        for radius, alpha in ((90, 24), (58, 46), (24, 90)):
            glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 170, 110, alpha), (radius, radius), radius)
            surface.blit(glow, (sun_x - radius, sun_y - radius))

        # Tiny moons to avoid a flat sky
        pygame.draw.circle(surface, (155, 132, 118), (int(w * 0.18), int(viewport_h * 0.38)), 7)
        pygame.draw.circle(surface, (105, 96, 112), (int(w * 0.25), int(viewport_h * 0.22)), 4)

        for sx, sy, size, brightness, phase in self.stars:
            twinkle = int(brightness * (0.7 + 0.3 * math.sin(self.anim_time * 2 + phase)))
            twinkle = max(0, min(255, twinkle))
            pygame.draw.circle(surface, (twinkle, twinkle, twinkle),
                               (sx, sy), max(1, int(size)))

        if len(self.terrain_points) > 2:
            pygame.draw.polygon(surface, (140, 70, 40), self.terrain_points)
            pygame.draw.lines(surface, (180, 100, 60), False,
                              self.terrain_points[:-2], 2)

        # Thin dust haze over the exterior horizon
        haze = pygame.Surface((w, viewport_h), pygame.SRCALPHA)
        for y in range(viewport_h):
            alpha = int(28 * max(0.0, 1.0 - y / max(viewport_h, 1)))
            pygame.draw.line(haze, (214, 118, 64, alpha), (0, y), (w, y))
        surface.blit(haze, (0, 0))

    def _draw_viewport_frame(self, surface):
        """Draw the upper greenhouse canopy frame framing the Martian exterior."""
        w, h = self.window_size
        viewport_h = h // 3

        outer = pygame.Rect(0, 0, w, viewport_h + 6)
        inner = pygame.Rect(14, 12, w - 28, viewport_h - 8)
        pygame.draw.rect(surface, self.VIEWPORT_SHADOW, outer)
        pygame.draw.rect(surface, self.VIEWPORT_METAL, inner, border_radius=8)
        pygame.draw.rect(surface, (152, 168, 160), inner.inflate(-6, -6), 3, border_radius=8)

        for x in (w // 3, 2 * w // 3):
            pygame.draw.rect(surface, (70, 76, 84), (x - 5, 16, 10, viewport_h - 18))
        for x in range(24, w - 24, 50):
            pygame.draw.circle(surface, (52, 56, 61), (x, 18), 3)
            pygame.draw.circle(surface, (52, 56, 61), (x, viewport_h + 1), 3)

        for x in range(44, w - 44, 88):
            pygame.draw.line(surface, (112, 126, 118), (x, 16), (x + 42, viewport_h - 10), 3)
            pygame.draw.line(surface, (82, 96, 90), (x + 10, 16), (x + 52, viewport_h - 10), 1)

        gloss = pygame.Surface((w, viewport_h), pygame.SRCALPHA)
        pygame.draw.polygon(
            gloss,
            (255, 255, 255, 18),
            [(18, 18), (w * 0.46, 18), (w * 0.24, viewport_h - 10), (18, viewport_h - 10)],
        )
        surface.blit(gloss, (0, 0))

    def _track_bounds_screen(self, track, pad_x=110, pad_y=90):
        pts = np.array([self.convert_pos(p) for p in track.centerline], dtype=float)
        min_x = max(26, int(np.min(pts[:, 0]) - pad_x))
        max_x = min(self.window_size[0] - 26, int(np.max(pts[:, 0]) + pad_x))
        viewport_h = self.window_size[1] // 3
        min_y = max(viewport_h + 24, int(np.min(pts[:, 1]) - pad_y))
        max_y = min(self.window_size[1] - 44, int(np.max(pts[:, 1]) + pad_y))
        return pygame.Rect(min_x, min_y, max(40, max_x - min_x), max(40, max_y - min_y))

    def _draw_hull_bay(self, surface, track):
        """Draw a greenhouse repair gantry around the breach zone."""
        bay = self._track_bounds_screen(track, pad_x=125, pad_y=105)
        outer = bay.inflate(80, 72)
        shadow = outer.move(10, 12)
        pygame.draw.rect(surface, (0, 0, 0, 76), shadow, border_radius=26)

        pygame.draw.rect(surface, self.BULKHEAD_CORE, outer, border_radius=24)
        pygame.draw.rect(surface, self.HULL_TRIM, outer, 3, border_radius=24)
        pygame.draw.rect(surface, self.BULKHEAD_INNER, outer.inflate(-18, -18), 2, border_radius=20)

        service = bay.inflate(34, 26)
        pygame.draw.rect(surface, (22, 26, 34), service, border_radius=18)
        pygame.draw.rect(surface, self.HULL_ACCENT_DK, service, 2, border_radius=18)
        recess = bay.inflate(-12, -8)
        for y in range(recess.top, recess.bottom):
            t = (y - recess.top) / max(1, recess.height)
            col = (
                int(32 + 18 * (1 - t)),
                int(38 + 20 * (1 - t)),
                int(46 + 24 * (1 - t)),
            )
            pygame.draw.line(surface, col, (recess.left, y), (recess.right, y))

        for x in (recess.left + 38, recess.centerx, recess.right - 38):
            pygame.draw.line(surface, (70, 78, 90), (x, recess.top + 16), (x, recess.bottom - 16), 2)
            pygame.draw.line(surface, (120, 132, 148), (x - 1, recess.top + 16), (x - 1, recess.bottom - 16), 1)
        for y in (recess.top + 28, recess.bottom - 28):
            pygame.draw.line(surface, (58, 66, 80), (recess.left + 18, y), (recess.right - 18, y), 3)
            pygame.draw.line(surface, (126, 138, 152), (recess.left + 18, y - 1), (recess.right - 18, y - 1), 1)

        bolt_points = []
        for x in range(outer.left + 24, outer.right - 20, 42):
            bolt_points.append((x, outer.top + 14))
            bolt_points.append((x, outer.bottom - 14))
        for y in range(outer.top + 28, outer.bottom - 24, 38):
            bolt_points.append((outer.left + 14, y))
            bolt_points.append((outer.right - 14, y))
        for bx, by in bolt_points:
            pygame.draw.circle(surface, (30, 34, 42), (bx, by), 4)
            pygame.draw.circle(surface, (144, 152, 164), (bx - 1, by - 1), 2)

        for side in ("left", "right"):
            x = outer.left + 18 if side == "left" else outer.right - 72
            module = pygame.Rect(x, recess.top + 34, 54, recess.height - 68)
            pygame.draw.rect(surface, (50, 56, 68), module, border_radius=10)
            pygame.draw.rect(surface, (116, 126, 138), module, 2, border_radius=10)
            for yy in range(module.top + 12, module.bottom - 12, 26):
                led_col = self.HULL_ACCENT if (yy // 26) % 2 == 0 else (240, 112, 84)
                pygame.draw.circle(surface, led_col, (module.centerx, yy), 4)
                pygame.draw.circle(surface, (255, 255, 255), (module.centerx - 1, yy - 1), 1)

        stripe_w = min(150, outer.width // 3)
        for side_x in (outer.left + 30, outer.right - stripe_w - 30):
            stripe = pygame.Rect(side_x, outer.top + 22, stripe_w, 16)
            pygame.draw.rect(surface, self.HULL_STRIPE, stripe, border_radius=3)
            for x in range(stripe.left, stripe.right, 14):
                pygame.draw.line(surface, self.HULL_STRIPE_DK, (x, stripe.bottom), (x + 10, stripe.top), 4)

        mono = pygame.font.SysFont("Consolas", 11, bold=True)
        labels = [
            ("GREENHOUSE REPAIR GANTRY", outer.left + 32, outer.bottom - 30, self.HUD_CYAN),
            ("BAY GH-2", outer.right - 104, outer.bottom - 30, (210, 214, 220)),
            ("PRESSURE LOSS DETECTED", outer.left + 32, outer.top + 48, (255, 172, 104)),
        ]
        for text, lx, ly, color in labels:
            surface.blit(mono.render(text, True, color), (lx, ly))

        for side in ("left", "right"):
            x0 = outer.left - 8 if side == "left" else outer.right + 8
            x1 = outer.left + 24 if side == "left" else outer.right - 24
            for i, col in enumerate(((88, 140, 168), (160, 170, 184), (190, 110, 84))):
                y = outer.centery - 26 + i * 22
                ctrl_x = x0 + (18 if side == "left" else -18)
                pygame.draw.lines(surface, col, False, [(x0, y), (ctrl_x, y - 10), (x1, y)], 4)

        glow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for x in (service.left + 4, service.right - 4):
            pygame.draw.line(glow, (90, 190, 225, 48), (x, service.top + 12), (x, service.bottom - 12), 6)
        for y in (service.top + 4, service.bottom - 4):
            pygame.draw.line(glow, (90, 190, 225, 26), (service.left + 12, y), (service.right - 12, y), 4)
        surface.blit(glow, (0, 0))

    def _draw_wall_surface(self, surface):
        """Draw the greenhouse shell, glazing and crop beds behind the crack."""
        w, h = self.window_size
        viewport_h = h // 3

        for y in range(viewport_h, h):
            t = (y - viewport_h) / max(h - viewport_h, 1)
            col = (
                int(self.MARS_WALL[0] * (1 - 0.18 * t) + self.MARS_WALL_LT[0] * 0.22),
                int(self.MARS_WALL[1] * (1 - 0.16 * t) + self.MARS_WALL_LT[1] * 0.18),
                int(self.MARS_WALL[2] * (1 - 0.14 * t) + self.MARS_WALL_LT[2] * 0.16),
            )
            pygame.draw.line(surface, col, (0, y), (w, y))

        glass_sheen = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for y in range(viewport_h + 10, h, 3):
            t = (y - viewport_h) / max(h - viewport_h, 1)
            alpha = max(0, int(30 * (1.0 - t)))
            pygame.draw.line(glass_sheen, (190, 228, 220, alpha), (18, y), (w - 18, y))
        surface.blit(glass_sheen, (0, 0))

        for sx, sy, sr, shade in self.wall_spots:
            if sy <= viewport_h:
                continue
            c = (
                max(0, min(255, self.MARS_WALL[0] + shade)),
                max(0, min(255, self.MARS_WALL[1] + shade)),
                max(0, min(255, self.MARS_WALL[2] + shade)),
            )
            pygame.draw.circle(surface, c, (sx, sy), sr)

        pygame.draw.line(surface, (78, 96, 86), (0, viewport_h), (w, viewport_h), 4)

        for x in range(28, w - 28, 96):
            rib = pygame.Rect(x, viewport_h + 8, 18, h - viewport_h - 22)
            pygame.draw.rect(surface, (80, 96, 88), rib, border_radius=4)
            pygame.draw.line(surface, (160, 176, 166), (rib.left + 3, rib.top), (rib.left + 3, rib.bottom), 2)

        for rect in self.wall_panels:
            shade = 8 if ((rect.x // 20 + rect.y // 20) % 2 == 0) else -3
            panel_col = (
                max(0, min(255, self.MARS_WALL_LT[0] + shade)),
                max(0, min(255, self.MARS_WALL_LT[1] + shade)),
                max(0, min(255, self.MARS_WALL_LT[2] + shade)),
            )
            glaze = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            glaze.fill((*panel_col, 148))
            surface.blit(glaze, rect.topleft)
            pygame.draw.rect(surface, self.MARS_WALL_DK, rect, 2, border_radius=6)
            inset = rect.inflate(-10, -10)
            pygame.draw.rect(surface, (86, 104, 98), inset, 1, border_radius=5)
            pygame.draw.line(surface, self.HULL_TRIM, (rect.left + 5, rect.top + 4), (rect.right - 5, rect.top + 4), 1)
            pygame.draw.line(surface, (54, 66, 62), (rect.left + 5, rect.bottom - 4), (rect.right - 5, rect.bottom - 4), 1)

        for x, y in self.wall_rivets:
            pygame.draw.circle(surface, (42, 34, 34), (x, y), 4)
            pygame.draw.circle(surface, (133, 112, 100), (x - 1, y - 1), 2)

        warning_font = pygame.font.SysFont("Consolas", 10, bold=True)
        for plate in self.warning_plates:
            pygame.draw.rect(surface, self.HULL_STRIPE, plate, border_radius=2)
            for x in range(plate.left, plate.right, 12):
                pygame.draw.line(surface, self.HULL_STRIPE_DK,
                                 (x, plate.bottom), (x + 8, plate.top), 3)
            txt = warning_font.render("GLAZING BREACH", True, (240, 236, 224))
            surface.blit(txt, (plate.x + 6, plate.y + 3))

        for vent in self.vent_slots:
            pygame.draw.rect(surface, (52, 68, 66), vent, border_radius=3)
            pygame.draw.rect(surface, (26, 29, 34), vent, 2, border_radius=3)
            for x in range(vent.left + 6, vent.right - 3, 10):
                pygame.draw.line(surface, (118, 126, 136), (x, vent.top + 3), (x, vent.bottom - 4), 1)

        for y in range(viewport_h + 52, h - 30, 144):
            pygame.draw.line(surface, (66, 74, 88), (24, y), (w - 24, y), 4)
            pygame.draw.line(surface, (126, 136, 148), (24, y - 2), (w - 24, y - 2), 1)
        for x in range(42, w - 42, 160):
            pygame.draw.rect(surface, (60, 68, 80), (x, viewport_h + 36, 26, h - viewport_h - 80), border_radius=6)

        bed_glow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for bed in self.crop_beds:
            pygame.draw.rect(surface, (72, 78, 64), bed, border_radius=5)
            pygame.draw.rect(surface, (112, 118, 96), bed, 2, border_radius=5)
            pygame.draw.rect(surface, (34, 42, 34), bed.inflate(-8, -8), border_radius=4)
            for stem_x in range(bed.left + 10, bed.right - 8, 10):
                stem_h = 10 + ((stem_x + bed.y) % 12)
                stem_top = bed.top + 8
                pygame.draw.line(surface, (84, 164, 82), (stem_x, bed.bottom - 8), (stem_x, bed.bottom - stem_h), 2)
                pygame.draw.ellipse(surface, (112, 198, 102), (stem_x - 4, stem_top + ((stem_x // 2) % 10), 8, 5))
            pygame.draw.rect(bed_glow, (110, 210, 128, 18), bed.inflate(8, 8), border_radius=8)
        surface.blit(bed_glow, (0, 0))

        streak_layer = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for sx, sy, length, alpha in self.wall_streaks:
            pygame.draw.line(streak_layer, (22, 24, 28, alpha), (sx, sy), (sx + 8, sy + length), 2)
            pygame.draw.line(streak_layer, (170, 176, 184, alpha // 4), (sx - 1, sy), (sx + 7, sy + length), 1)
        surface.blit(streak_layer, (0, 0))

        vignette = pygame.Surface(self.window_size, pygame.SRCALPHA)
        pygame.draw.rect(vignette, (0, 0, 0, 0), (0, 0, w, h))
        pygame.draw.rect(vignette, (0, 0, 0, 54), (0, viewport_h, w, h - viewport_h), 12)
        surface.blit(vignette, (0, 0))

    def _spawn_leak_particles(self, track):
        """Emit particles from exposed crack sections to suggest atmosphere loss."""
        if len(self.leak_particles) >= self.max_leak_particles:
            return

        count = min(4, self.max_leak_particles - len(self.leak_particles))
        for _ in range(count):
            idx = random.randint(6, max(6, track.n_pts - 7))
            pt = track.centerline[idx]
            normal = track.normals[idx]
            sx, sy = self.convert_pos(pt)
            outward = normal if random.random() > 0.5 else -normal
            speed = random.uniform(16.0, 52.0)
            vx = outward[0] * speed * 0.65 + random.uniform(-12.0, 12.0)
            vy = -abs(outward[1] * speed) - random.uniform(8.0, 30.0)
            color = self.LEAK_COLOR if random.random() > 0.18 else self.LEAK_HOT
            self.leak_particles.append(
                LeakParticle(
                    sx,
                    sy,
                    vx,
                    vy,
                    random.uniform(0.45, 1.35),
                    random.randint(1, 3),
                    color,
                )
            )

    def _update_leak_particles(self, dt):
        alive = []
        for particle in self.leak_particles:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vx *= 0.995
            particle.vy -= 3.8 * dt
            particle.life -= dt
            if particle.life > 0:
                alive.append(particle)
        self.leak_particles = alive

    def _draw_leak_particles(self, surface):
        for particle in self.leak_particles:
            alpha = max(0, min(255, int(255 * particle.life / max(particle.max_life, 1e-6))))
            radius = particle.size * 2
            sprite = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(sprite, (*particle.color, alpha), (radius, radius), radius)
            surface.blit(sprite, (int(particle.x) - radius, int(particle.y) - radius))

    def _draw_dust_overlay(self, surface, dt):
        """Floating internal dust enhances scale and depth around the work area."""
        dust_layer = pygame.Surface(self.window_size, pygame.SRCALPHA)
        w, h = self.window_size
        for mote in self.dust_motes:
            mote[0] = (mote[0] + mote[3] * dt) % (w + 20) - 10
            mote[1] += mote[4] * dt
            if mote[1] < h // 3 + 12:
                mote[1] = h - 12
            elif mote[1] > h - 10:
                mote[1] = h // 3 + 18
            alpha = int(38 + 42 * (0.5 + 0.5 * math.sin(self.anim_time + mote[5])))
            pygame.draw.circle(
                dust_layer,
                (220, 208, 190, alpha),
                (int(mote[0]), int(mote[1])),
                int(mote[2]),
            )
        surface.blit(dust_layer, (0, 0))

    def _draw_mars_track(self, surface, track, highlight_wall=None):
        """Render the path as a glowing crack in the greenhouse glazing."""
        center_pts = [self.convert_pos(p) for p in track.centerline]
        left_pts = [self.convert_pos(p) for p in track.wall_left]
        right_pts = [self.convert_pos(p) for p in track.wall_right]
        lip_left = [self.convert_pos(p + n * 0.0014) for p, n in zip(track.centerline, track.normals)]
        lip_right = [self.convert_pos(p - n * 0.0014) for p, n in zip(track.centerline, track.normals)]
        scorch_left = [self.convert_pos(p + n * 0.0044) for p, n in zip(track.centerline, track.normals)]
        scorch_right = [self.convert_pos(p - n * 0.0044) for p, n in zip(track.centerline, track.normals)]

        scorch = pygame.Surface(self.window_size, pygame.SRCALPHA)
        scorch_poly = [(int(p[0]), int(p[1])) for p in scorch_left] + [(int(p[0]), int(p[1])) for p in reversed(scorch_right)]
        if len(scorch_poly) > 5:
            pygame.draw.polygon(scorch, (*self.SCORCH, 44), scorch_poly)
            pygame.draw.polygon(scorch, (230, 108, 72, 12), scorch_poly, 3)
        surface.blit(scorch, (0, 0))

        shadow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for i in range(len(center_pts) - 1):
            p1 = (int(center_pts[i][0] + 4), int(center_pts[i][1] + 4))
            p2 = (int(center_pts[i + 1][0] + 4), int(center_pts[i + 1][1] + 4))
            pygame.draw.line(shadow, (18, 10, 9, 72), p1, p2,
                             max(4, int(track.half_width * self.window_scale * 2.2)))
        surface.blit(shadow, (0, 0))

        glow_surf = pygame.Surface(self.window_size, pygame.SRCALPHA)
        if len(center_pts) > 1:
            glow_width = int(track.half_width * self.window_scale * 4.2)
            for i in range(len(center_pts) - 1):
                pulse = 0.7 + 0.3 * math.sin(self.anim_time * 4 + i * 0.05)
                alpha = int(92 * pulse)
                color = (*self.CRACK_GLOW, alpha)
                p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
                p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
                pygame.draw.line(glow_surf, color, p1, p2, glow_width)
        surface.blit(glow_surf, (0, 0))

        if len(lip_left) > 1:
            pygame.draw.lines(surface, (210, 226, 220), False, [(int(p[0]), int(p[1])) for p in lip_left], 2)
            pygame.draw.lines(surface, (42, 26, 22), False, [(int(p[0]), int(p[1])) for p in right_pts], 3)
        if len(lip_right) > 1:
            pygame.draw.lines(surface, (210, 226, 220), False, [(int(p[0]), int(p[1])) for p in lip_right], 2)
            pygame.draw.lines(surface, (42, 26, 22), False, [(int(p[0]), int(p[1])) for p in left_pts], 3)

        for i in range(len(center_pts) - 1):
            p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
            p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
            pygame.draw.line(surface, (26, 12, 10), p1, p2, 6)
            pygame.draw.line(surface, self.CRACK_CORE, p1, p2, 4)
            pygame.draw.line(surface, self.CRACK_HOT, p1, p2, 1)

            if i % 26 == 0 and i < len(track.normals):
                base = np.array(center_pts[i], dtype=float)
                tangent = np.array(track.centerline[min(i + 1, track.n_pts - 1)]) - np.array(track.centerline[max(i - 1, 0)])
                if np.linalg.norm(tangent) > 1e-8:
                    tangent = tangent / np.linalg.norm(tangent)
                    tangent_s = np.array([-tangent[0], tangent[1]])
                else:
                    tangent_s = np.array([0.0, 1.0])
                normal = np.array(track.normals[i])
                branch_dir = normal * (1 if (i // 26) % 2 == 0 else -1) + 0.35 * tangent_s
                if np.linalg.norm(branch_dir) > 1e-8:
                    branch_dir = branch_dir / np.linalg.norm(branch_dir)
                branch_len = 12 + (i % 3) * 7
                end = base + branch_dir * branch_len
                pygame.draw.line(surface, (120, 55, 42), base.astype(int), end.astype(int), 2)
                pygame.draw.line(surface, (246, 118, 72), base.astype(int), end.astype(int), 1)

        left_color = (255, 50, 50) if highlight_wall == 'left' else (160, 80, 50)
        right_color = (255, 50, 50) if highlight_wall == 'right' else (160, 80, 50)

        if len(left_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in left_pts]
            pygame.draw.lines(surface, left_color, False, pts, 2)
        if len(right_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in right_pts]
            pygame.draw.lines(surface, right_color, False, pts, 2)

        start_s = self.convert_pos(track.start)
        end_s = self.convert_pos(track.end)

        for r in [14, 10, 6]:
            alpha = int(80 * (14 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 255, 100, alpha + 60), (r, r), r)
            surface.blit(s, (int(start_s[0]) - r, int(start_s[1]) - r))
        pygame.draw.circle(surface, (0, 255, 120), (int(start_s[0]), int(start_s[1])), 5)

        for r in [14, 10, 6]:
            alpha = int(80 * (14 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 60, 60, alpha + 60), (r, r), r)
            surface.blit(s, (int(end_s[0]) - r, int(end_s[1]) - r))
        pygame.draw.circle(surface, (255, 80, 80), (int(end_s[0]), int(end_s[1])), 5)

        label_font = pygame.font.SysFont("Consolas", 11, bold=True)
        surface.blit(label_font.render("START", True, (0, 255, 120)),
                     (int(start_s[0]) + 14, int(start_s[1]) - 5))
        surface.blit(label_font.render("END", True, (255, 80, 80)),
                     (int(end_s[0]) + 14, int(end_s[1]) - 5))

    def _draw_work_lamp(self, surface, cursor_phys):
        """Local task lighting centered on the operator tool."""
        if cursor_phys is None:
            return
        cursor = self.convert_pos(cursor_phys)
        x, y = int(cursor[0]), int(cursor[1])
        lamp = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for radius, alpha in ((168, 8), (118, 14), (72, 24)):
            pygame.draw.circle(lamp, (232, 242, 255, alpha), (x, y), radius)
        # Slight elongated reflection on the metal bay
        pygame.draw.ellipse(lamp, (214, 234, 255, 18), (x - 82, y + 18, 164, 62))
        pygame.draw.ellipse(lamp, (255, 190, 138, 20), (x - 56, y + 12, 112, 34))
        surface.blit(lamp, (0, 0))

    def _draw_mission_hud(self, surface, state_name, n_demos, crack_name, progress, elapsed):
        """Draw a greenhouse mission HUD overlay on the VR panel."""
        w = self.window_size[0]
        hud_font = pygame.font.SysFont("Consolas", 13)
        title_font = pygame.font.SysFont("Consolas", 16, bold=True)

        bar_h = 28
        bar_surf = pygame.Surface((w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 180))
        surface.blit(bar_surf, (0, 0))

        title = title_font.render("MARS GREENHOUSE — AUTONOMOUS CRACK SEAL", True, self.HUD_ORANGE)
        surface.blit(title, (10, 5))

        status_text = f"[{state_name}]  Crack: {crack_name}  Demos: {n_demos}"
        status = hud_font.render(status_text, True, (180, 180, 180))
        surface.blit(status, (w - status.get_width() - 10, 7))

        # Side diagnostics stack
        diag = pygame.Surface((180, 82), pygame.SRCALPHA)
        diag.fill((0, 0, 0, 120))
        surface.blit(diag, (w - 194, 48))
        metrics = [
            ("Pressure retention", f"{int(100 * progress):02d}%"),
            ("Leak severity", f"{int(100 * (1.0 - progress)):02d}%"),
            ("Crop bay status", "STABLE" if progress > 0.35 else "AT RISK"),
        ]
        for i, (label, value) in enumerate(metrics):
            ly = 56 + i * 22
            surface.blit(hud_font.render(label, True, (142, 148, 154)), (w - 184, ly))
            color = self.HUD_CYAN if i < 2 else ((130, 255, 160) if progress > 0.35 else (255, 108, 82))
            surface.blit(hud_font.render(value, True, color), (w - 74, ly))

        bottom_y = self.window_size[1] - 30
        bot_surf = pygame.Surface((w, 30), pygame.SRCALPHA)
        bot_surf.fill((0, 0, 0, 160))
        surface.blit(bot_surf, (0, bottom_y))

        info_parts = [
            f"Progress: {progress * 100:.0f}%",
            f"Mission: {progress * 100:.0f}%",
            f"Time: {elapsed:.1f}s",
        ]
        info_text = hud_font.render("   |   ".join(info_parts), True, (200, 200, 200))
        surface.blit(info_text, (10, bottom_y + 7))

    def _draw_seal_progress_bar(self, surface, seal_pct, x, y, w_bar=200, h_bar=16):
        """Draw the Mars mission progress bar."""
        pygame.draw.rect(surface, (24, 27, 33), (x - 4, y - 4, w_bar + 8, h_bar + 8), border_radius=5)
        pygame.draw.rect(surface, (30, 30, 40), (x, y, w_bar, h_bar), border_radius=3)

        fill_w = int(w_bar * seal_pct)
        if fill_w > 0:
            r = int(40 * (1 - seal_pct))
            g = int(100 + 155 * seal_pct)
            b = int(200 * (1 - seal_pct * 0.5))
            pygame.draw.rect(surface, (r, g, b), (x, y, fill_w, h_bar), border_radius=3)

        pygame.draw.rect(surface, (100, 100, 120), (x, y, w_bar, h_bar), 1, border_radius=3)

        bar_font = pygame.font.SysFont("Consolas", 11)
        label = bar_font.render(f"SEAL: {seal_pct * 100:.0f}%", True, (220, 220, 220))
        surface.blit(label, (x + w_bar // 2 - label.get_width() // 2, y + 1))

    def draw_mars_vr_scene(self, track, state_name, n_demos, track_name,
                           progress, elapsed, highlight_wall=None, cursor_phys=None):
        """Render the Mars greenhouse crack environment on the VR panel."""
        dt = 1.0 / max(self.FPS, 1)
        self.anim_time += dt
        self._draw_space_background(self.screenVR)
        self._draw_viewport_frame(self.screenVR)
        self._draw_wall_surface(self.screenVR)
        self._draw_porthole(self.screenVR)
        self._draw_hull_bay(self.screenVR, track)
        self._draw_mars_track(self.screenVR, track, highlight_wall=highlight_wall)
        self._spawn_leak_particles(track)
        self._update_leak_particles(dt)
        self._draw_leak_particles(self.screenVR)
        self._draw_dust_overlay(self.screenVR, dt)
        self._draw_work_lamp(self.screenVR, cursor_phys)
        self._draw_seal_progress_bar(
            self.screenVR, progress, self.window_size[0] - 220, self.window_size[1] // 3 + 40
        )
        self._draw_mission_hud(
            self.screenVR, state_name, n_demos, track_name, progress, elapsed
        )

    def erase_screen(self):
        self.screenHaptics.fill(self.cWhite) #erase the haptics surface
        self.screenVR.fill((0, 0, 0)) #erase the VR surface
        self.debug_text = ""
    
    def render(self,pA0,pB0,pA,pB,pE,f,pM):
        ###################Render the Haptic Surface###################
        #set new position of items indicating the endpoint location
        self.haptic.center = pE #the hhandle image and effort square will also use this position for drawing
        self.effort_cursor.center = self.haptic.center

        if self.device_connected:
            self.effort_color = (255,255,255)

        #pygame.draw.rect(self.screenHaptics, self.effort_color, self.haptic,border_radius=4)
        pygame.draw.rect(self.screenHaptics, self.effort_color, self.effort_cursor,border_radius=8)

        ######### Robot visualization ###################
        if self.show_linkages:
            pantographColor = (150,150,150)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pA0,pA],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pB0,pB],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pA,pE],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pB,pE],15)
            
            for p in ( pA0,pB0,pA,pB,pE):
                pygame.draw.circle(self.screenHaptics, (0, 0, 0),p, 15)
                pygame.draw.circle(self.screenHaptics, (200, 200, 200),p, 6)
        
        ### Hand visualisation
        self.screenHaptics.blit(self.hhandle,self.effort_cursor)
        
        #pygame.draw.line(self.screenHaptics, (0, 0, 0), (self.haptic.center),(self.haptic.center+2*k*(xm-xh)))
        
        ###################Render the VR surface###################
        if not self.hide_haptic_vr:
            pygame.draw.rect(self.screenVR, self.colorHaptic, self.haptic, border_radius=8)
        
        if not self.device_connected:
            pygame.draw.lines(self.screenHaptics, (0,0,0), False,[self.effort_cursor.center,pM],2)
        ##Fuse it back together
        self.window.blit(self.screenHaptics, (0,0))
        self.window.blit(self.screenVR, (self.window_size[0],0))

        ##Print status in  overlay
        if self.show_debug:    
            self.debug_text += "FPS = " + str(round(self.clock.get_fps()))+" "
            self.debug_text += "fe: "+str(np.round(f[0],1))+","+str(np.round(f[1],1))+"] "
            self.debug_text += "xh: ["+str(np.round(pE[0],1))+","+str(np.round(pE[1],1))+"]"
            self.text = self.font.render(self.debug_text, True, (0, 0, 0), (255, 255, 255))
            self.window.blit(self.text, self.textRect)

        pygame.display.flip()    
        ##Slow down the loop to match FPS
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()
