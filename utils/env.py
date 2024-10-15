import math
import numpy as np
import copy

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [DotDict(item) if isinstance(item, dict) else item for item in value]
            self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __deepcopy__(self, memo):
        # Check if this object is already in the memo dictionary
        if id(self) in memo:
            return memo[id(self)]
        
        # Create a new instance of DotDict
        new_dict = DotDict({})
        
        # Save the new instance in the memo dictionary before deep copying attributes
        memo[id(self)] = new_dict
        
        # Deep copy each attribute
        for key, value in self.__dict__.items():
            new_dict.__dict__[key] = copy.deepcopy(value, memo)
        
        return new_dict
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __rich_repr__(self):
        for key, value in self.__dict__.items():
            yield key, value

class History:
    def __init__(self, max_length):
        self.obs_history = []
        self.max_length = max_length

    def add(self, item):
        self.obs_history.append(item)
        if len(self.obs_history) > self.max_length:
            self.obs_history.pop(0)

    def get(self):
        return self.obs_history

def is_goal(ball, goal_size=750):
    return ball[0] > 4500 and ball[1] < goal_size and ball[1] > -goal_size

def is_ball_in_opp_half(ball):
    return ball[0] > 0

def get_unit_vector(a, b):
    diff_x = b[0] - a[0]
    diff_y = b[1] - a[1]
    magnitude = math.sqrt(diff_x**2 + diff_y**2)
    return { 'x': diff_x / magnitude, 'y': diff_y / magnitude }

def get_angle(a, b):
    '''
    Returns the angle between two points.
    '''
    return np.arctan2(b[1] - a[1], b[0] - a[0])

def is_between(self, a, b, c, tol=400):
    '''
    Checks if (b) is between (a) and (c).
    '''

    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])
    
    ab = b - a
    ac = c - a
    
    # proj. of ac onto ab
    proj = (ac @ ab / ab @ ab) * ab
    
    # vec. from c to the proj. on the line
    c_to_proj = ac - proj
    return np.linalg.norm(c_to_proj) < tol

def dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def is_facing(a, b, tol=10):
    '''
    Checks if point a is facing point b.
    '''
    if a[0] == b[0] and a[1] == b[1]:
        return False

    a_angle = math.degrees(a[2]) % 360

    # find the angle between a and b
    a_b_angle = math.degrees(math.atan2(
        b[1] - a[1],
        b[0] - a[0],
    ))

    # check if the a is facing the b
    angle = (a_angle - a_b_angle) % 360
    return angle < tol or angle > 360 - tol

def is_out_of_bounds(a, tol=0, goal_size=750, goal_is_out=True):
    '''
    
    '''
    if abs(a[0]) > 4500 + tol or abs(a[1]) > 3000 + tol:
        in_goal = abs(a[0]) > 4500 and abs(a[1]) > goal_size

        if in_goal and not goal_is_out:
            return False
        else:
            return True
    
    return False


def is_out_of_field(a, tol=0):
    '''
    
    '''
    return abs(a[0]) > 5200 + tol or abs(a[1]) > 3700 + tol

def get_rel_obs(a, b, noise=0):
    '''
    Returns the pose of B relative to A assuming 
    A = [x, y, theta] and
    B = [x, y, ...]
    '''

    b = [
        b[0] + np.random.normal(0, noise),
        b[1] + np.random.normal(0, noise)
    ]

    delta_x = b[0] - a[0]
    delta_y = b[1] - a[1]

    angle = np.arctan2(delta_y, delta_x) - a[2]
    new_x = delta_x * np.cos(-a[2]) - delta_y * np.sin(-a[2])
    new_y = delta_x * np.sin(-a[2]) + delta_y * np.cos(-a[2])
    return [new_x / 10000, new_y / 10000, np.sin(angle), np.cos(angle)]

def can_kick(agent, ball, tol=200):
    return is_facing(agent, ball) and dist(agent, ball) < tol
    
def normalize_angle(angle):
    '''
    Normalizes angle to [-pi, pi].
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

def kick_ball(agent, ball, strength, tol=200):
    # if robot is close enough to ball, kick ball
    if can_kick(agent, ball, tol=tol):
        ball[2] = agent[2]
        ball[3] = 70 * strength + 20