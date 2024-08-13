import numpy as np
import math
import matplotlib.pyplot as plt
from cells import *
from random import Random

random = Random()

symbols = {
    0 : '↑',
    1 : '↓',
    2 : '←',
    3 : '→'
}

class Board():

    def __init__(self, rows_no, cols_no):
        cells = self.generate_cells(rows_no, cols_no)
        rows, cols, cells = self.process_cells(cells)
        self.rows_no = rows
        self.cols_no = cols
        self.cells = cells
        self.mouse_row = None
        self.mouse_col = None
        
        # Saves all moves made by the agent in the explotation faze for visualization purposes
        self.moves = []

        self.value_texts = []
        self.action_texts = []

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(left=0, right=self.rows_no)
        self.ax.set_ylim(top=self.cols_no, bottom=0)
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(0, rows, step=1))
        self.ax.set_yticks(np.arange(cols-1, -1, step=-1))

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        x, y = event.xdata, event.ydata

        if x is not None and y is not None:
            self.mouse_row = math.floor(y)
            self.mouse_col = math.floor(x)
        
    def generate_cells(self, rows_no, cols_no) -> list[list[Cell]]:
        # Distribution of regular, penalty, wall and teleport cells
        cell_codes = np.random.choice(4, size=(rows_no, cols_no), p=[0.65, 0.15, 0.15, 0.05])
        # A terminal cell is placed in a random place on the board
        cell_codes[random.randint(0, rows_no-1), random.randint(0, cols_no-1)] = 4

        cells = [[self.int_to_cell(cell_codes[i, j], rows_no, cols_no) for i in range(rows_no)] for j in range(cols_no)]
        return cells

    def process_cells(self, cells: list[list[Cell]]):
        cells = [list(row) for row in cells]

        if not cells:
            raise Exception("Number of rows in a board must be at least one.")
        if not cells[0]:
            raise Exception("There has to be at least one column.")

        rows_no = len(cells)
        cols_no = len(cells[0])
        for row in cells:
            if not row or len(row) != cols_no:
                raise Exception(
                    "Each row in a a board must have the same number of columns. ")
        
        return rows_no, cols_no, cells

    def __getitem__(self, key: tuple[int, int]) -> Cell:
        r, c = key
        return self.cells[r][c]

    def int_to_cell(self, code: int, rows_no : int, cols_no : int) -> Cell:
        if code == 0:
            return RegCell(-1)
        elif code == 1:
            return RegCell(-10)
        elif code == 2:
            return WallCell()
        elif code == 3:
            return TelCell(Position(random.randint(0, rows_no-1), random.randint(0, cols_no-1)))
        elif code == 4:
            return TermCell(0)

    def hide_text(self, text):
        text.set_visible(False)
    
    def show_text(self, text):
        text.set_visible(True)

    def draw_action_values(self, q_values):
        [self.hide_text(text) for text in self.value_texts]
        [self.hide_text(text) for text in self.action_texts]
        
        s = Position(self.mouse_row, self.mouse_col)

        try:
            if s in q_values and not self[s.row, s.col].is_teleport():
                value_dict = q_values[s]
                
                if 0<=s.row-1<self.rows_no and value_dict['UP'] is not None:
                    text = self.ax.text(s.col+0.4, s.row-1+0.75, str(f"{value_dict['UP']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
                if 0<=s.row+1<self.rows_no and value_dict['DOWN'] is not None:
                    text = self.ax.text(s.col+0.4, s.row+1+0.75, str(f"{value_dict['DOWN']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
                if 0<=s.col-1< self.cols_no and value_dict['LEFT'] is not None:
                    text = self.ax.text(s.col-1+0.4, s.row+0.75, str(f"{value_dict['LEFT']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
                if 0<=s.col+1<self.cols_no and value_dict['RIGHT'] is not None:
                    text = self.ax.text(s.col+1+0.4, s.row+0.75, str(f"{value_dict['RIGHT']:.1f}"), fontsize=10)
                    if text not in self.action_texts:
                        self.action_texts.append(text)
            elif self[s.row, s.col].is_teleport():
                sn = self[s.row, s.col].get_next_cell()
                print(f"Q_values: {q_values[s]}")
                text = self.ax.text(sn.col+0.4, sn.row+0.75, str(f"{q_values[s]['UP']:.1f}"), fontsize=10)
                if text not in self.action_texts:
                    self.action_texts.append(text)
        except Exception as e:
            print(e)
            
    def draw_values(self, values: dict[Position : float]):
        if len(self.value_texts):
            [self.hide_text(text) for text in self.value_texts]
        else:
            for s in values:                
                text = self.ax.text(s.col+0.4, s.row+0.75, str(f"{values[s]:.1f}"), fontsize=10)
                self.value_texts.append(text)

    def draw_board(self):
        board_img = np.ones(shape=(self.rows_no, self.cols_no, 3), dtype=np.uint8)
        teleport_index = 0
        for i in range(self.rows_no):
            for j in range(self.cols_no):
                if isinstance(self[i, j], RegCell):
                    if self[i, j].get_reward() == -1:
                        board_img[i, j, :] = [255, 255, 255] # Regular cell
                    else:
                        board_img[i, j, :] = [255, 0, 0] # Regular cell with penalty
                elif isinstance(self[i, j], WallCell):
                    board_img[i, j, :] = [0, 0, 0] # Wall cell
                elif isinstance(self[i, j], TelCell):
                    board_img[i, j, :] = [0, 255, 0] # Teleport cell

                    self.ax.text(j+0.1, i+0.2, str(teleport_index), color="pink", fontweight="bold", fontsize=10)
                    print(f"Teleport {teleport_index}: {i}, {j}")
                    next_cell = self[i,j].get_next_cell()
                    self.ax.text(next_cell.col+0.1, next_cell.row+0.2, str(teleport_index),color="pink",  fontweight="bold", fontsize=10)
                    print(f"Teleport to {teleport_index}: {next_cell.row}, {next_cell.col}")

                    teleport_index += 1
                else:
                    board_img[i, j, :] = [0, 0, 255] # Terminal cell

        self.ax.imshow(board_img, extent=(0, self.cols_no, self.rows_no, 0), origin="upper")

    def draw_agent(self, pos=(0,0), avatar="+"):
        row, col = pos
        text = self.ax.text(col+0.4, row+0.6, avatar, fontweight="bold", color="orange", fontsize=30)
        text.set_zorder(10)
        self.moves.append(text)

        #Determines the duration of each step
        plt.pause(0.5)
        text.set_visible(False)
