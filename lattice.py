import numpy as np
import Cell
import math

class Lattice:
    def __init__(self, N, T, adhesion_list, targets):
        self.T = T
        self.N = N
        self.lattice = np.full((N,N), fill_value=2, dtype=int)  # all background
        self.adhesion_list = adhesion_list
        self.targets = targets
        self.cells = {}   # cid -> Cell

    def monte_carlo_step(self):
        for _ in range(self.N * self.N):
            self.random_copying_sample()
        for cell in self.cells.values():
            cell.full_calc_perimeter(self.lattice, self.N, start=True)

    def add_cell(self, cid, ctype, positions):
        """Manually seed a cell into the lattice"""
        cell = Cell.Cell(cid, ctype)
        for (i,j) in positions:
            # Only add positions that are within bounds
            if 0 <= i < self.N and 0 <= j < self.N:
                self.lattice[i,j] = cid
                cell.add_site((i,j))
        cell.full_calc_perimeter(self.lattice, self.N, True)
        self.cells[cid] = cell

    def total_adhesion(self):
        total = 0
        for row, col in np.ndindex(self.lattice.shape):
            cur = self.lattice[row,col]
            if cur == 2:  # background
                t_cur = 2
            else:
                t_cur = self.cells[cur].ctype
            
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                ni, nj = row + di, col + dj
                
                # Check if neighbor is within bounds
                if 0 <= ni < self.N and 0 <= nj < self.N:
                    neighbor = self.lattice[ni, nj]
                    if neighbor == 2:
                        t_nei = 2
                    else:
                        t_nei = self.cells[neighbor].ctype
                else:
                    # Out of bounds - treat as background (type 2)
                    t_nei = 2
                
                if t_cur != t_nei:
                    total += self.adhesion_list[t_cur][t_nei]
        return total
    
    def random_copying_sample(self):
        sx, sy = np.random.randint(0, self.N, 2)
        nei_list = [(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)]
        rix = np.random.randint(0, len(nei_list))
        dx, dy = nei_list[rix]

        tx = sx + dx
        ty = sy + dy

        # Check if target is within bounds
        if 0 <= tx < self.N and 0 <= ty < self.N:
            if self.lattice[tx][ty] != self.lattice[sx][sy]:
                self.try_to_copy(sx, sy, tx, ty)
        # If target is out of bounds, do nothing (no copying across boundaries)
        return
    
    def try_to_copy(self, sx, sy, tx, ty):
        delta_H = self.calculate_delta_H(sx, sy, tx, ty)
        if delta_H <= 0:
            accept_copy = True
        else:
            P_copy = math.exp((-delta_H)/self.T)
            accept_copy = np.random.random() < P_copy
        
        if accept_copy:
            # Get the source cell that's expanding
            source_cid = self.lattice[sx, sy]
            target_cid = self.lattice[tx, ty]
            # Update the target cell if it's not background (losing a site)
            if target_cid != 2:  # not background
                target_cell = self.cells[target_cid]
                target_cell.remove_site((tx, ty))
                target_cell.update_perimeter((tx, ty), self.lattice, self.N, adding=False, hypothetical=False)
            # Update the lattice
            self.lattice[tx, ty] = source_cid
            
            if source_cid != 2:
                # Update the source cell (gaining a site)
                source_cell = self.cells[source_cid]
                source_cell.add_site((tx, ty))
                self.lattice[tx][ty] = source_cid
                source_cell.update_perimeter((tx, ty), self.lattice, self.N, adding=True, hypothetical=False)

    def calculate_delta_H(self, sx, sy, tx, ty):
        delta_H_vol = 0
        delta_H_per = 0
        
        # Handle source cell (if not background)
        if self.lattice[sx][sy] != 2:
            s_cell = self.cells[self.lattice[sx][sy]]
            delta_H_vol += self.delta_H_volume_for_single_cell(s_cell, 1)
            
            s_perim_change = s_cell.update_perimeter((tx, ty), self.lattice, self.N, 
                                                    hypothetical=True, adding=True)
            delta_H_per += self.delta_H_perimeter_for_single_cell(s_cell, s_perim_change)
        
        # Handle target cell (if not background)  
        if self.lattice[tx][ty] != 2:
            t_cell = self.cells[self.lattice[tx][ty]]
            delta_H_vol += self.delta_H_volume_for_single_cell(t_cell, -1)
            
            t_perim_change = t_cell.update_perimeter((tx, ty), self.lattice, self.N,
                                                    hypothetical=True, adding=False)
            delta_H_per += self.delta_H_perimeter_for_single_cell(t_cell, t_perim_change)

        # Delta adhesion calculation with fixed boundaries
        delta_H_adh = 0
        matrix_old = np.full((3, 3), fill_value=2, dtype=int)  # Initialize with background
        matrix_new = np.full((3, 3), fill_value=2, dtype=int)  # Initialize with background
        
        for i in range(3):
            for j in range(3):
                actual_x = tx - 1 + i
                actual_y = ty - 1 + j
                
                # Only fill matrix if coordinates are within bounds
                if 0 <= actual_x < self.N and 0 <= actual_y < self.N:
                    matrix_old[i][j] = self.lattice[actual_x][actual_y]
                    matrix_new[i][j] = self.lattice[actual_x][actual_y]
                # Out of bounds cells remain as background (2)
        
        matrix_new[1][1] = self.lattice[sx][sy]
        
        old_adhesion = self.calculate_local_adhesion(matrix_old)
        new_adhesion = self.calculate_local_adhesion(matrix_new)
        delta_H_adh = new_adhesion - old_adhesion
        
        return delta_H_vol + delta_H_per + delta_H_adh

    def delta_H_volume_for_single_cell(self, c_cell, volume_change):
        """Calculate volume constraint energy change for a single cell
        
        Args:
            c_cell: The cell object
            volume_change: +1 if cell gains a site, -1 if cell loses a site
        """
        c_cell_vt, c_cell_vs = self.targets[c_cell.ctype][0]  # target volume, strength
        
        old_volume = c_cell.volume
        new_volume = old_volume + volume_change
        
        # Energy = λ * (volume - target)²
        old_energy = c_cell_vs * (old_volume - c_cell_vt)**2
        new_energy = c_cell_vs * (new_volume - c_cell_vt)**2
        
        return new_energy - old_energy
        
    def delta_H_perimeter_for_single_cell(self, c_cell, perimeter_change):
        c_cell_pt, c_cell_ps = self.targets[c_cell.ctype][1]
        # Similar to volume: λ * (new_perimeter² - old_perimeter²)
        old_perim = c_cell.perimeter
        new_perim = old_perim + perimeter_change
        return c_cell_ps * (new_perim - c_cell_pt)**2 - c_cell_ps * (old_perim - c_cell_pt)**2

    def calculate_local_adhesion(self, matrix):
        """Calculates the local adhesion in a 3x3 matrix"""
        local_adhesion = 0
        for x in range(3):
            for y in range(3):
                c_type = self.cells[matrix[x][y]].ctype if matrix[x][y] != 2 else 2
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    ni, nj = x + di, y + dj
                    # Check bounds within the 3x3 matrix
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        nei = matrix[ni][nj]
                        nei_type = self.cells[nei].ctype if nei != 2 else 2
                        if c_type != nei_type:
                            local_adhesion += self.adhesion_list[c_type][nei_type]
        return local_adhesion