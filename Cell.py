class Cell:
    def __init__(self, cid, ctype):
        self.cid = cid  # unique cell ID
        self.ctype = ctype  # cell type (e.g., 0=bg, 1=red, 2=black)
        self.sites = set()  # {(i,j), ...}
        self.volume = 0
        self.perimeter = 0

    def add_site(self, pos):
        self.sites.add(pos)
        self.volume += 1

    def remove_site(self, pos):
        self.sites.remove(pos)
        self.volume -= 1

    def full_calc_perimeter(self, lattice, N, start):
        """Expensive: full recompute of perimeter with fixed boundaries"""
        perim = 0
        for i, j in self.sites:
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:  # 8-neigh
                ni, nj = i + di, j + dj
                
                # Check if neighbor is within bounds
                if 0 <= ni < N and 0 <= nj < N:
                    if lattice[ni][nj] != self.cid:
                        perim += 1
                else:
                    # Out of bounds neighbor - counts as different (background)
                    perim += 1
                    
        if start:
            self.perimeter = perim
        return perim

    def full_calc_volume(self):
        self.volume = len(self.sites)

    def update_perimeter(self, pos, lattice, N, hypothetical, adding):
        """Update perimeter when adding or removing a site with fixed boundaries"""
        i, j = pos
        delta = 0
        
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:  # 8-neigh
            ni, nj = i + di, j + dj
            
            # Check if neighbor is within bounds
            if 0 <= ni < N and 0 <= nj < N:
                if lattice[ni, nj] == self.cid:
                    # internal neighbor
                    delta -= 1 if adding else +1
                else:
                    # neighbor is different cell
                    delta += 1 if adding else -1
            else:
                # Out of bounds neighbor - always counts as external boundary
                delta += 1 if adding else -1
                
        if not hypothetical:
            self.perimeter += delta
        return delta