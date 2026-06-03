# Input states for Agent step function
from grid_adventure.grid import GridState
from grid_adventure.grid import State
from grid_adventure.grid import to_state
from grid_adventure.actions import Action
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.env import ImageObservation
# State steppers
from grid_adventure.step import Action
from grid_adventure.step import step
from grid_adventure.grid import step as grid_step
from grid_adventure.entities import (
    AgentEntity,
    FloorEntity,
    WallEntity,
    ExitEntity,
    CoinEntity,
    GemEntity,
    KeyEntity,
    LockedDoorEntity,
    UnlockedDoorEntity,
    LavaEntity,
    BoxEntity,
    SpeedPowerUpEntity,
    ShieldPowerUpEntity,
    PhasingPowerUpEntity,
)
# Utility helpers
from queue import PriorityQueue
from collections import deque

import os
import numpy as np
from PIL import Image

import json

with open("weight.json", "r") as f:
    data = json.load(f)

#background weights
bg_w = data["bg_w"]
bg_i = data["bg_i"]

#content item weights
ct_w = data["ct_w"]
ct_i = data["ct_i"]

class Agent:
    """Grid Adventure: Variant 1 agent template.
    This class is the single public interface that Coursemology will import and
    interact with when evaluating your submission. You should extend the
    internals (add helper classes / functions in other files if you wish) but
    MUST preserve:
    1. The class name: Agent
    2. The public method: step(self, state: GridState | ImageObservation) -> Action
    High‑level lifecycle per environment tick:
        state  --->  step(...)  --->  Action
    The "state" object type depends on the task:
    - Task 1: A fully structured GridState instance.
    - Task 2: An ImageObservation dictionary whose primary observation is an RGBA image
      plus limited structured metadata in the 'info' sub‑dict. In this case you
      typically perform perception to build (or approximate) an internal
      structured representation before planning.
    - Task 3: Input state could be either a GridState instance 
      or an ImageObservation dictionary
    Constraints:
    - Keep per‑step latency small (single CPU, ~1GB RAM). Avoid O(W*H) scans of
      the full grid every step.
    - Determinism helps reproducibility; seed your own RNG if you add any
      random components.
    You may add __init__ parameters (with defaults) if needed for your own
    development, but the grader will instantiate Agent() with no arguments.
    """
    
    def __init__(self):
        self.active = False

        self.dx = [0, 0, -1, 1, 0]; self.dy = [-1, 1, 0, 0, 0];
        self.basicActions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        self.total_gem = 0; self.total_lock = 999
        self.defaultpos = (0,0)
        self.targetpos = (0,0)

        self.w = 2.5
        self.PC = 0 # program counter to avoid cmp isequal
        self.index = -1; self.len = 0
        self.result = []
        self.bfs = [[]]; self.nodoorbfs = [[]]

        self.noMoveGrid = None

        self.bgw = np.array(bg_w); self.bgi = np.array(bg_i)
        self.bg_output = [FloorEntity, LavaEntity, WallEntity]

        self.ctw = np.array(ct_w); self.cti = np.array(ct_i)
        self.ct_output = [FloorEntity, AgentEntity, ExitEntity, BoxEntity, LockedDoorEntity, GemEntity, KeyEntity, CoinEntity, SpeedPowerUpEntity, ShieldPowerUpEntity, PhasingPowerUpEntity]

        self.pickable = [CoinEntity, GemEntity, KeyEntity, SpeedPowerUpEntity, ShieldPowerUpEntity, PhasingPowerUpEntity]
        """Initialize your agent.
        Put all one‑time setup here (e.g., hardcoded ML model weights, 
        precomputing heuristic tables). Keep it fast and memory‑light 
        to respect platform limits.
        """
        # Placeholder for any future initialization logic
        pass
    
    def extract_features(self, tile_pixels):
        tile_rgb = tile_pixels[:, :, :3]
        img = Image.fromarray(tile_pixels[:, :, :3].astype(np.uint8))
        tile_rgb = np.array(img.resize((48, 48)))

        bg_feat = np.concatenate([tile_rgb[:1, :, :], tile_rgb[-1:, :, :]], axis=0).flatten()

        shredded_ct = tile_rgb[5:-5, 5:-5, :]
        
        h, w, c = shredded_ct.shape
        size = 3
        new_h, new_w = h // size, w // size

        reshaped = shredded_ct[:new_h*size, :new_w*size, :].reshape(new_h, size, new_w, size, c)
        pooled_ct = reshaped.mean(axis=(1, 3))
        
        ct_feat = pooled_ct.flatten() # 147 features

        return bg_feat, ct_feat

    def toGrid(self, img: ImageObservation):
        image = img["image"]
        #i = Image.fromarray(img_obs["image"])
        #display(i)  
        grid_w = img["info"]["config"]["width"]
        grid_h = img["info"]["config"]["height"]
        h = img["info"]["agent"]["health"]["current_health"]

        pixel_h, pixel_w, _ = image.shape
        tile_h = pixel_h // grid_h
        tile_w = pixel_w // grid_w

        grid_info = []

        for x in range(grid_w):
            row_info = []
            x_start = int(round(x * pixel_w / grid_w))
            x_end = int(round((x + 1) * pixel_w / grid_w))

            for y in range(grid_h):
                y_start = int(round(y * pixel_h / grid_h))
                y_end = int(round((y + 1) * pixel_h / grid_h))
                
                tile = image[y_start:y_end, x_start:x_end, :]

                tile_info = [FloorEntity()]

                #i = Image.fromarray(tile)
                #display(i) 
                #  
                #background
                bg_feat, ct_feat = self.extract_features(tile)

                bg_idx = np.argmax(np.dot(self.bgw, bg_feat) + self.bgi)
                bg_entity = self.bg_output[bg_idx]()
                
                ct_entity = FloorEntity()
                if isinstance(bg_entity, FloorEntity):
                    ct_idx = np.argmax(np.dot(self.ctw, ct_feat) + self.cti)
                    ct_entity = self.ct_output[ct_idx]()
                
                if not isinstance(bg_entity, FloorEntity):
                    tile_info.append(bg_entity)
                    #print(bg_entity)
                if not isinstance(ct_entity, FloorEntity):
                    tile_info.append(ct_entity)
                    #print(ct_entity)

                row_info.append(tile_info)

            grid_info.append(row_info)

        gridstate = GridState(width=grid_w, height=grid_h, movement=MOVEMENTS["cardinal"], objective=OBJECTIVES["collect_gems_and_exit"])
        for x in range(grid_w):
            for y in range(grid_h):    
                for e in grid_info[x][y]:
                    if isinstance(e, AgentEntity):
                        e.set_health(h)
                        gridstate.add((x, y), e)
                    else:
                        gridstate.add((x,y), e)
        return gridstate
    
    # -----------

    def bfsdistance(self, pos, state: GridState):
        self.bfs = [[999 for _ in range(state.height)] for _ in range(state.width)]
        q = deque(); q.append(pos); self.bfs[pos[0]][pos[1]] = 0
        while len(q) > 0:
            x,y = q.popleft(); 
            for i in range(4):
                newX = x + self.dx[i]; newY = y + self.dy[i]
                if (not 0 <= newX < state.width) or (not 0 <= newY < state.height):
                    continue
                if self.bfs[newX][newY] < 999:
                    continue
                entity = state.objects_at((newX, newY))
                wallBlocked = False
                for e in entity:
                    if isinstance(e, WallEntity) or isinstance(e, LockedDoorEntity):
                        wallBlocked = True
                        self.bfs[newX][newY] = 1000
                        break
                if wallBlocked:
                    continue
                self.bfs[newX][newY] = self.bfs[x][y] + 1
                q.append((newX, newY))
        # --------    
        self.nodoorbfs = [[999 for _ in range(state.height)] for _ in range(state.width)]
        q = deque(); q.append(pos); self.nodoorbfs[pos[0]][pos[1]] = 0
        while len(q) > 0:
            x,y = q.popleft(); 
            for i in range(4):
                newX = x + self.dx[i]; newY = y + self.dy[i]
                if (not 0 <= newX < state.width) or (not 0 <= newY < state.height):
                    continue
                if self.nodoorbfs[newX][newY] < 999:
                    continue
                entity = state.objects_at((newX, newY))
                wallBlocked = False
                for e in entity:
                    if isinstance(e, WallEntity):
                        wallBlocked = True
                        self.nodoorbfs[newX][newY] = 1000
                        break
                if wallBlocked:
                    continue
                self.nodoorbfs[newX][newY] = self.nodoorbfs[x][y] + 1
                q.append((newX, newY))

    def heur(self, gem, key, ghostinv, speedinv, pos):
        #idk whether need improvement
        x,y = pos; targetX, targetY = self.targetpos
        
        haveghost = False; speedscale = 1; havekey = False
        if speedinv >= 1:
            speedscale = 2
        if ghostinv >= 1:
            haveghost = True
        if key >= 1:
            havekey = True

        pathlen = 0
        if haveghost: 
            pathlen = (abs(x - targetX) + abs(y - targetY)) / speedscale
        elif (not haveghost) and self.bfs[x][y] >= 999:
            pathlen = (abs(x - targetX) + abs(y - targetY)) / speedscale
        elif (not haveghost) and havekey: 
            pathlen = (self.nodoorbfs[x][y]) / speedscale
        else: 
            pathlen = (self.bfs[x][y]) / speedscale 
        
        return pathlen # + 2 * (self.total_gem - gem)

    # -------------
    def keycount(self, state: State, agentID):
        inventory = state.inventory.get(agentID)
        return sum(1 for eid in inventory.item_ids if eid in state.key)
    
    def coincount(self, state: State, agentID):
        inventory = state.inventory.get(agentID)
        return sum(1 for eid in inventory.item_ids if eid in state.rewardable)
    
    def gemcount(self, state: State, agentID):
        inventory = state.inventory.get(agentID)
        return sum(1 for eid in inventory.item_ids if eid in state.requirable)
    
    def speedcount(self, state: State, agentID):
        status = state.status.get(agentID)
        return sum(1 for eid in status.effect_ids if eid in state.speed)
    
    def ghostcount(self, state: State, agentID):
        status = state.status.get(agentID)
        return sum(1 for eid in status.effect_ids if eid in state.phasing)

    def shieldcount(self, state: State, agentID):
        status = state.status.get(agentID)
        return sum(1 for eid in status.effect_ids if eid in state.immunity)

    def lockcount(self, state: State):
        return len(list(state.locked.keys()))

    def canpick(self, pos, state: State):
        for cid in state.collectible.keys():
            p = state.position.get(cid)
            if p.x == pos[0] and p.y == pos[1]:
                return True
        return False
    
    def canusekey(self, pos, state: State):
        x,y = pos
        nearby = []
        for i in range(4):
            newX = x + self.dx[i]; newY = y + self.dy[i]
            if (not 0 <= newX < state.width) or (not 0 <= newY < state.height):
                continue
            nearby.append((newX, newY))
        
        havelockeddoor = False
        for did in state.locked.keys():
            if havelockeddoor:
                break
            p = state.position.get(did)
            for pp in nearby:
                if pp[0] == p.x and pp[1] == p.y:
                    havelockeddoor = True
                    break

        return havelockeddoor

    def doorlocked(self, pos, state: State):
        x,y = pos
        for kid in state.locked.keys():
            p = state.position.get(kid)
            if p.x == x and p.y == y:
                return True
        return False

    def update(self, act, key, coin, pos, state: State):
        change = 0
    
        new_state = state
        
        try:
            new_state = step(state, act)
        except Exception as e: # TimeoutException may be caught here
            if e.__class__.__name__ == "TimeoutException": raise e
        

        agent_id = next(iter(new_state.agent.keys()))
        p = new_state.position.get(agent_id);  
        new_pos = (p.x, p.y)

        if new_state.lose:
            return [0, 0, 0, 999999999, 0, 0, 0, pos, state] # key = 999999999 API for failure
        
        #update data

        new_gem = self.gemcount(new_state, agent_id)
        new_key = self.keycount(new_state, agent_id)
        new_coin = self.coincount(new_state, agent_id)
        new_lock = self.lockcount(new_state)


        new_ghost = self.ghostcount(new_state, agent_id)
        new_speed = self.speedcount(new_state, agent_id)
        new_shield = self.shieldcount(new_state, agent_id)

        #print(act, new_gem, new_key, new_coin, new_lock, new_ghost, new_speed, new_shield)

        change += 3
        if act == Action.PICK_UP and new_coin == coin + 1:
            change -= 5

        #ret
        return [change, new_gem, new_lock, new_key, new_shield, new_ghost, new_speed, new_pos, new_state]
        # old version backup [change, new_gem, new_unlock, new_key, new_shield, new_ghost, new_speed, new_pos, new_state]
        
    def nxt(self, pos, state: State):
        agent_id = next(iter(state.agent.keys()))
        possible_moves = [] #possible_moves = [Action.WAIT]
        x,y = pos

        #check movement PICK_UP
        pick = self.canpick(pos, state)
        if pick:
            possible_moves = [Action.PICK_UP] + possible_moves
            
        #check movement basic movements
        ghost = True if self.ghostcount(state, agent_id) > 0 else False
        for i in range(4):
            newX = x + self.dx[i]; newY = y + self.dy[i]
            if (not 0 <= newX < state.width) or (not 0 <= newY < state.height):
                continue

            entity = self.noMoveGrid.objects_at((newX, newY))
            wallBlocked = False
            for e in entity:
                if isinstance(e, WallEntity):
                    wallBlocked = True
                    break
            if self.doorlocked((newX, newY), state):
                wallBlocked = True
            if ghost:
                wallBlocked = False
            if not wallBlocked:
                possible_moves.append(self.basicActions[i])

        #check movement use key
        useKey = self.canusekey(pos, state)
        if useKey == True:
            possible_moves = [Action.USE_KEY] + possible_moves            

        #end
        return possible_moves #+ [Action.WAIT]
        
    def Astar(self, gridstate: GridState):
        self.noMoveGrid = gridstate
        state = to_state(gridstate)
        agent_id = next(iter(state.agent.keys()))

        visited = dict()
        pq = PriorityQueue()

        self.active = True
        agent_id = next(iter(state.agent.keys()))
        dpos = state.position.get(agent_id); self.defaultpos = (dpos.x, dpos.y) 

        exit_id = next(iter(state.exit.keys()))
        tpos = state.position.get(exit_id); self.targetpos = (tpos.x, tpos.y)

        self.total_gem = self.gemcount(state, agent_id)
        self.total_lock = self.lockcount(state)

        self.bfsdistance(self.targetpos, gridstate)

        defaultHeur = self.heur(0, 0, 0, 0, self.defaultpos)
        defaultStats = (-1 * 0, -1 * self. w * 3 * defaultHeur, self.total_lock, -1 * 0, -1 * 0, -1 * 0, -1 * 0, -1 * 0, self.defaultpos, self.PC, state, [])
        # -gem, heur_cost, lock, -key, -shield, -ghost, -speed, -coin, pos, PC, state, path
        
        pq.put( defaultStats )
        while not pq.empty():
            gem, heur_cost, lock, key, shield, ghost, speed, coin, pos, _, s, path = pq.get()
            gem *= -1; key *= -1; shield *= -1; ghost *= -1; speed *= -1; coin *= -1

            cost = heur_cost + self.w * 3 * self.heur(gem, key, ghost, speed, pos)

            # print screen debug
            #display(renderer.render(s))

            #ensure no revisit
            id = (gem, lock, key, shield, ghost, speed, pos)
            
            if id in visited and cost >= visited[id]:
                continue
            visited[id] = cost
            
            # iter next steps
            next_step = self.nxt(pos, s)

            for act in next_step:
                change, new_gem, new_lock, new_key, count_shield, count_ghost, count_speed, new_pos, new_state \
                    = self.update(act, key, coin, pos, s)
                
                new_shield = 1 if count_shield >= 1 else 0
                new_ghost = 1 if count_ghost >= 1 else 0
                new_speed = 1 if count_speed >= 1 else 0

                if new_key == 999999999:
                    continue
                
                new_heur = self.heur(new_gem, new_key, new_ghost, new_speed, new_pos)
                new_cost = cost + change 
                newpath = path + [act]
                new_coin = coin

                if change < 3:
                    new_coin += 1
                if new_state.win == True:
                    self.result = newpath
                    return None
                if new_state.turn > 150:
                    continue
                
                #no revisit
                new_id = (new_gem, new_lock, new_key, new_shield, new_ghost, new_speed, new_pos)
                if new_id in visited and new_cost >= visited[new_id]:
                    continue
                
                #print(act, cost, new_cost, self.w * new_heur * 3, -1 * (new_cost + self.w * 3 * new_heur))

                self.PC += 1
                pq.put([-1 * new_gem, new_cost - self.w * 3 * new_heur, new_lock, -1 * new_key, -1 * new_shield, -1 * new_ghost, -1 * new_speed, -1 * new_coin, new_pos, self.PC, new_state, newpath])
        return None
    
    def step(self, state: GridState | ImageObservation) -> Action:
        if self.active == False:
            grid_state = None
            if isinstance(state, dict) and "image" in state:
                grid_state = self.toGrid(state)
            else:
                grid_state = state

            self.Astar(grid_state)
            self.len = len(self.result)
            print("Result:", self.result) 

        self.index += 1
        if self.index < self.len:
            return self.result[self.index]
        else:
            self.__init__()
            return Action.WAIT
        #safety
        return Action.WAIT
   
agent = Agent()    
'''
env = create_env(build_level_required_multiple, observation_type='image')
img_obs, _ = env.reset()
action = agent.step(img_obs)

'''
agent.step(gridstate_2_3)

