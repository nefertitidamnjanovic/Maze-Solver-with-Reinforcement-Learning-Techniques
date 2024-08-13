import numpy as np
import matplotlib.pyplot as plt
from random import Random 
from threading import Event
from itertools import chain, repeat
import copy
from board import Board, symbols
from cells import (
    Cell,
    RegCell, 
    Position, 
    Actions
)

random = Random()
running = Event()

class Environment():
    '''
    Attributes: 
        - board   : generated board that defines all cells
        - states  : all steppable steps available to the agent
        - actions : all actions defined by Action enum'''

    def __init__(self, board : Board):
        self.board = board
        self.states = self.init_states()
        self.actions = self.init_actions()

    def update_state_value(self, s:Position, v:dict[Position:float], gamma:float, actions) -> float:
        '''Returns the maximum state value for the given state'''
        values = []
        for a in actions:
            sn, r = self.next_state(s, a)
            if sn is not None:
                values.append(r + gamma*v[sn])

        # If no action was possible in the current state it's state value is given the lowest number
        return max(values) if len(values) else v[s]-11

    def update_all_state_values(self, v : dict[Position:float], gamma:float, policy:dict) \
        -> dict[Position:float]:
        '''Returns a new set of state-values calculated by using the given policy.
        If value iteration is being used, policy contains all available actions.
        If policy iteration is being used, policy contains the action defined by the current policy'''

        # Original state-values are not changed
        values = copy.deepcopy(v)
        for s in self.states:
            if not self.board[s()].is_terminal():
                actions = [action for action in self.actions if action in policy[s]]

                if self.board[s()].is_teleport():
                    sn = self.board[s()].get_next_cell()
                    actions = [action for action in self.actions if action in policy[sn]]

                values[s] = self.update_state_value(s, v, gamma, actions)

        return values

    def update_action_value(self, s : Position, a : Actions, q : dict[Position:dict[Actions:float]], \
        gamma : float, policy : dict) -> float:
        '''Returns the maximum state-action value for the given state and action'''

        q_values = []
        sn, r = self.next_state(s, a)
        if sn is None:
            return None
        elif self.board[sn()].is_terminal():
            return 0

        actions = [action for action in self.actions if action in policy[sn]]

        for an in actions:
            if q[sn][an] is None:
                continue

            q_values.append(r + gamma*q[sn][an]) 

        # If no action was possible None is returned
        return max(q_values) if len(q_values) else None

    def update_all_action_values(self, q : dict[Position:dict[Actions:float]], \
        gamma:float, policy:dict) -> dict[Position:dict[Actions:float]]:
        '''Returns a new set of state-action values calculated by using the given policy.
        If value iteration is being used, policy contains all available actions.
        If policy iteration is being used, policy contains the action defined by the current policy'''

        # Original q-values are not changes
        q_values = copy.deepcopy(q)
        for s in self.states:
            if not self.board[s()].is_terminal():
                for a in self.actions:
                    q_values[s][a] = self.update_action_value(s, a, q, gamma, policy)
        return q_values

    def next_state(self, s : Position, a : Actions) -> tuple[Position, int]:
        '''Returns the next state and reward given the current state and action'''

        # If the agent is on a teleport cell it is actually on the cell that it will be teleported on
        if self.board[s()].is_teleport():
            s = self.board[s()].get_next_cell()
         
        if a == Actions.UP.name:
            s_new = Position(s.row-1, s.col)
        elif a == Actions.DOWN.name:
            s_new = Position(s.row+1, s.col)
        elif a == Actions.LEFT.name:
            s_new = Position(s.row, s.col-1)
        elif a == Actions.RIGHT.name:
            s_new = Position(s.row, s.col+1)

        if 0 <= s_new.row < self.board.rows_no and \
            0 <= s_new.col < self.board.cols_no and \
            self.board[s_new()].is_steppable():

            if self.board[s_new()].is_teleport():
                s_new = self.board[s_new()].get_next_cell()

                # TODO proveriti da li moze da se desi
                # if not self.board[s_new()].is_steppable():
                #     return None, None

            return s_new, self.board[s_new()].get_reward()

        # If the given action is not possible in the current state, None is returnded
        return None, None
        
    def init_actions(self) -> dict[str: int]:
        '''Returns a dictionary of all possible actions'''

        actions = {}
        for a in Actions:
            actions[a.name] = a.value

        return actions

    def init_states(self) -> list[Position]:
        '''Returns a list of all possible states the agent can get to.
        Processes cells for teleports.'''

        states = []
        for r in range(self.board.rows_no):
            for c in range(self.board.cols_no):
                pos = Position(r, c)
                if not self.board[pos()].is_steppable():
                    continue

                states.append(pos)

                if self.board[pos()].is_teleport():
                    next_cell = self.board[pos()].get_next_cell()

                    # The reward for the teleport cell is the same as for the 
                    # cell it leads to, if it's possible to teleport to it
                    if self.board[next_cell()].is_steppable() and pos != next_cell:
                        self.board[pos()].set_reward(self.board[next_cell()].get_reward())
                    else:
                        # Teleports that lead into walls are treated as regular cells
                        self.board.cells[pos.row][pos.col] = RegCell(-1)
                        print(f"Teleport on ({pos.row}, {pos.col}) was removed.")
        return states

def get_error(new:dict, old:dict) -> float:
    '''Returns the maximum error between items of old and new state, or state-action, values'''
    
    # new and old have identical keys, old is an updated version of new
    key = next(iter(new))

    if isinstance(new[key], (int, float)) or new[key] is None:
        new = {k: (0 if v is None else v) for k, v in new.items()}
        old = {k: (0 if v is None else v) for k, v in old.items()}
        err = max([abs(new[x] - old[x]) for x in new])
        return err

    # Recursion occurs in case of state-action values
    max_val = []
    for s in new:
        max_val.append(get_error(new[s], old[s]))

    return max(max_val)

def value_iteration(update:callable, values:dict, gamma:float, eps:float, iterations:int = 100, policy:dict = None) \
    -> dict[Position:float]:
    '''Performs value iteration algorithm until all state values converge or until the given number of iterations passes'''

    new_values = copy.deepcopy(values)
    old_values = copy.deepcopy(values)
    for k in range(iterations):
        new_values = update(old_values, gamma, policy)

        err = get_error(copy.deepcopy(new_values), copy.deepcopy(old_values))

        old_values = new_values

        if err < eps:
            print(f"Value iteration finished in {k} iterations")
            return new_values, None

    # State values didn't converge
    print(f"Value exploration cut off")
    return new_values, None

def policy_iteration(update:callable, values:dict, policy:dict, update_policy:callable, gamma:float, eps:float, iterations:int=100) \
    -> dict[Position:float]:
    '''Performs policy iteration algorithm until the policy converges or until the given number of iterations passes'''

    new_values = copy.deepcopy(values)
    for k in range(iterations):
        # Each iteration is started from the same values passed to the function
        # Only the policy changes between iterations
        old_values = copy.deepcopy(values)
        new_values, _ = value_iteration(update, old_values, gamma, eps, iterations, policy)

        new_policy = {s:update_policy(env, s, new_values, gamma).name for s in env.states}
        
        if policy == new_policy:
            print(f"Policy iterations finished in {k} iterations")
            return policy, new_values

        policy = new_policy

    # Policy didn't converge
    print(f"Policy exploration cut off")
    return policy, new_values

def greedy(env:Environment, s:Position, v:dict[Position:float], gamma:float) -> Actions:
    '''Returns greedy action based on passed state values'''

    values = []
    min_v = min(v.values())

    for a in env.actions:
        s_new, r = env.next_state(s, a)

        if s_new != None and r != None:
            values.append(r + gamma * v[s_new])
        else:
            # If an action is impossible it needs to be added to the list so all actions 
            # remain on the same indexes
            values.append(min_v - 1000)

    return Actions(np.argmax(values))

def greedy_q(env:Environment, s:Position, q:dict[Position:dict[Actions:float]], gamma:float) -> Actions:
    '''Returns greedy action based on passed state-action values'''

    values = [(q[s][a], a) for a in env.actions if q[s][a] is not None]

    # If there are no possible actions in current state a random one is returned
    # Consequence of using policy iteration where the action dictated by the policy is impossible
    if not len(values):
        actions = list(env.actions.keys())
        return Actions(env.actions[random.choice(actions)])

    max_value = max(values, key=lambda x: x[0])

    return Actions(env.actions[max_value[1]])

def apply_policy(env:Environment, policy:callable, s:Position, gamma:float, values:dict[Position:float], pi:dict = None) \
    -> float:
    '''Simulates an episode using the given policy, state or state-action values and initial state
    Parameters:
        - policy : when using state values policy=greedy and is used to fetch the greedy action from current state
        - values : state values or state-action values
        - pi     : calculated policy when using policy iteration'''
    
    gain = 0
    i = 0

    while not board[s()].is_terminal() and running.is_set():
        if pi is not None:
            a = pi[s]
        else:
            a = policy(env, s, values, gamma).name

        s, r = env.next_state(s, a)
        gain += (gamma**i)*r
        i += 1
        symbol = symbols[env.actions[a]]

        board.ax.set_title(f"Gamma={gamma}, Gain={gain:.1f}")
        board.draw_agent(s(), symbol)

    return gain

def in_range(s:str, rows_n:int, cols_n:int) -> bool:
    '''Checks whether the given initial state is on the board'''
    try:
        numbers = s.split(",")
        r = int(numbers[0])
        c = int(numbers[1])

        if 0<=r<rows_n and 0<=c<cols_n:
            return True
    except Exception as e:
        print("Wrong format")
    
    return False

def get_start_pos(cells:list[list[Cell]], rows_no:int, cols_no:int) -> tuple[int, int]:
    '''Validates initial position given by the user'''

    prompts = chain(["Input starting position (r,c): "], \
            repeat(f"Rows must be in range (0 - {rows_no}) \nColumns must be in range (0 - {cols_no}) \nChosen cell must be steppable \nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: in_range(s, rows_no, cols_no) and cells[int(s[0])][int(s[2])].is_steppable(), replies))
    
    return int(valid[0]), int(valid[2])

def get_iteration_method() -> int:
    '''Validates method chosen by the user'''

    prompts = chain(["1. Value iteration\n2. Policy iteration\nChoose a iteration method: "], \
                    repeat(f"Acceptable answers are '1' or '2'.\nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: int(s) in [1, 2], replies))
    return int(valid)

def get_simulation_values() -> int:
    '''Validates method chosen by the user'''

    prompts = chain(["1. State-values\n2. Q-values\nChoose which values are used: "], \
                    repeat(f"Acceptable answers are '1' or '2'.\nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: s.isnumeric() and int(s) in [1, 2], replies))
    return int(valid)

def get_board() -> Board:
    '''Generates a board with dimensions chosen by the user'''

    prompts = chain(["Input board dimensions (r,c): "], repeat("Row and column numbers have to positive integers \nTry again: "))
    replies = map(input, prompts)

    valid = next(filter(lambda s: in_range(s, 100, 100), replies))
    valid = valid.split(",")
    r, c = int(valid[0]), int(valid[1])

    # Allows the user to generate boards until they are satisfied
    while True:
        board = Board(r, c)
        board.ax.set_title("Board preview")
        board.draw_board()

        if continue_question("Do you want to continue? (y/n): "):
            return board
        
        plt.close()

def continue_question(msg:str) -> bool:
    prompts = chain([msg], \
                    repeat(f"Only 'y' or 'n' are valid responses. \n{msg}"))
    replies = map(input, prompts)

    valid = next(filter(lambda s: s == 'y' or s == 'n', replies))

    print("------------------------------")

    if valid == 'y':
        return True
    return False

if __name__ == "__main__":
    running.set()

    plt.ion()
    board = get_board()

    env = Environment(board)

    gamma = float(input("Input gamma: "))
    eps = float(input("Input eps tolerance: "))

    # Initializes state and state-action values randomly
    v = {s:env.board[s()].get_reward() for s in env.states}
    q = {}
    for s in env.states:
        if not board[s()].is_terminal():
            q[s] = {}
            for a in env.actions:
                sn, _ = env.next_state(s,a)
                if sn is not None:
                    q[s][a] = (-10)*random.random()
                else:
                    q[s][a] = None
        else:
            q[s] = {a: None for a in env.actions}

    mode = get_iteration_method()

    actions = list(env.actions.keys())

    if mode == 1:
        # When value iteration is used all actions are possible
        policy = {s: actions for s in env.states if not board[s()].is_terminal()}
    
        values, policy_v = value_iteration(env.update_all_state_values, v, gamma, eps, policy=policy)
        q_values, policy_q = value_iteration(env.update_all_action_values, q, gamma, eps, policy=policy)
    elif mode == 2:
        # When policy iteraton is used only actions defined by the policy are possible
        policy = {s: random.choice(actions) for s in env.states if not board[s()].is_terminal()}

        policy_v, values = policy_iteration(env.update_all_state_values, v, policy, greedy, gamma, eps)
        policy_q, q_values = policy_iteration(env.update_all_action_values, q, policy, greedy_q, gamma, eps)

    board.ax.clear()
    board.ax.set_title("Board values preview")
    board.draw_board()
    board.draw_values(values)
    plt.pause(0.1)

    print("Click on a cell to view it's action values.")

    while 1:
        res = plt.waitforbuttonpress(8)
        if res is None:
            if not continue_question("Do you want to continue? (y/n): "):
                continue
            break
        
        if board.mouse_col is not None and board.mouse_row is not None:
            board.draw_action_values(q_values)
    
    [board.hide_text(text) for text in board.value_texts]
    [board.hide_text(text) for text in board.action_texts]

    try:
        while running.is_set():
            val = get_simulation_values()

            # Resets the board
            [board.hide_text(text) for text in board.moves]
            board.moves.clear()

            r, c = get_start_pos(board.cells, board.rows_no, board.cols_no)
            s = Position(r,c)

            board.ax.set_title(f"Gamma={gamma}, Gain={0}")
            board.draw_agent(s(), "+")

            # State values
            if val == 1:
                gain = apply_policy(env, greedy, s, gamma, values, policy_v)
            # State-action values
            elif val == 2:
                gain = apply_policy(env, greedy_q, s, gamma, q_values, policy_q)

            # Shows all moves made in this run
            [board.show_text(text) for text in board.moves]

            print("Gain: " + str(gain)) 

            if not continue_question("Do you want to play again? (y/n): "):
                running.clear()

    except KeyboardInterrupt:
        running.clear()

    plt.close('all')
    print("EXITED")

