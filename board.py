"""
Board representation for Catan.

Implements the standard Catan board layout with 19 hexagonal tiles,
resource assignment, number tokens, and precomputed quality matrices.
Uses exact vertex-to-tile and vertex-to-neighbor mappings.
"""

import random
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from quality import compute_quality


class Board:
    """
    Represents a Catan board with tiles, vertices, and precomputed quality scores.
    Uses exact mappings for vertices to tiles and vertex neighbors.
    """
    
    # Standard Catan resource distribution
    RESOURCE_COUNTS = {
        'wood': 4,
        'brick': 3,
        'wheat': 4,
        'ore': 3,
        'sheep': 4,
        'desert': 1
    }
    
    # Standard Catan number token distribution
    NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    
    # Exact vertex to tiles mapping (vertex_id: [tile_ids])
    VERTEX_TO_TILES = {
        0: [0, 1], 1: [0, 1, 4], 2: [0, 3, 4], 3: [0, 3], 4: [0], 5: [0],
        6: [1, 2], 7: [1, 2, 5], 8: [1, 4, 5], 9: [1], 10: [2], 11: [2, 6],
        12: [2, 5, 6], 13: [2],
        14: [3, 4, 8], 15: [3, 7, 8], 16: [3, 7], 17: [3],
        18: [4, 5, 9], 19: [4, 8, 9],
        20: [5, 6, 10], 21: [5, 9, 10], 22: [6], 23: [6, 11], 24: [6, 10, 11],
        25: [7, 8, 12], 26: [7, 12], 27: [7], 28: [7],
        29: [8, 9, 13], 30: [8, 12, 13],
        31: [9, 10, 14], 32: [9, 13, 14],
        33: [10, 11, 15], 34: [10, 14, 15], 35: [11], 36: [11], 37: [11, 15],
        38: [12, 13, 16], 39: [12, 16], 40: [12],
        41: [13, 14, 17], 42: [13, 16, 17],
        43: [14, 15, 18], 44: [14, 17, 18], 45: [15], 46: [15, 18],
        47: [16, 17], 48: [16], 49: [16],
        50: [17, 18], 51: [17], 52: [18], 53: [18]
    }
    
    # Exact vertex neighbors mapping (for distance rule)
    VERTEX_NEIGHBORS = {
        0: [1, 5, 9], 1: [0, 2, 8], 2: [1, 3, 14], 3: [2, 4, 17], 4: [3, 5], 5: [0, 4],
        6: [7, 9, 13], 7: [6, 8, 12], 8: [1, 7, 18], 9: [0, 6],
        10: [11, 13], 11: [10, 12, 22], 12: [7, 11, 20], 13: [6, 10],
        14: [2, 15, 19], 15: [14, 16, 25], 16: [15, 17, 28], 17: [3, 16],
        18: [8, 19, 21], 19: [14, 18, 29], 20: [12, 21, 24], 21: [18, 20, 31],
        22: [11, 23], 23: [22, 24, 35], 24: [20, 23, 33],
        25: [15, 26, 30], 26: [25, 27, 40], 27: [26, 28], 28: [16, 27],
        29: [19, 30, 32], 30: [25, 29, 38], 31: [21, 32, 34], 32: [29, 31, 41],
        33: [24, 34, 37], 34: [31, 33, 43], 35: [23, 36], 36: [35, 37], 37: [33, 36, 45],
        38: [30, 39, 42], 39: [38, 40, 49], 40: [26, 39],
        41: [32, 42, 44], 42: [38, 41, 47], 43: [34, 44, 46], 44: [41, 43, 50],
        45: [37, 46], 46: [43, 45, 52], 47: [42, 48, 51], 48: [47, 49], 49: [39, 48],
        50: [44, 51, 53], 51: [47, 50], 52: [46, 53], 53: [50, 52]
    }
    
    def __init__(self, seed: Optional[int] = None, num_players: int = 4,
                 quality_weights: Optional[Dict[str, float]] = None,
                 tile_config: Optional[List[Dict]] = None):
        """
        Initialize a Catan board.

        Args:
            seed: Random seed for reproducibility (ignored when tile_config is provided)
            num_players: Number of players (default: 4)
            quality_weights: Dictionary with keys 'w_resources', 'w_expected_cards', 'w_prob_at_least_one'
                           If None, uses default values (1/3 each)
            tile_config: List of 19 dicts, each with 'resource' (str) and 'number' (int or None).
                        If provided, uses this exact configuration instead of random assignment.
                        Resources: 'wood', 'brick', 'wheat', 'ore', 'sheep', 'desert'
                        Numbers: 2-12 (None for desert)
                        Example: [{'resource': 'wood', 'number': 6}, ..., {'resource': 'desert', 'number': None}]
        """
        if tile_config is None and seed is not None:
            random.seed(seed)

        self.num_players = num_players

        # Set quality weights
        if quality_weights is None:
            self.quality_weights = {
                'w_resources': 1/3,
                'w_expected_cards': 1/3,
                'w_prob_at_least_one': 1/3
            }
        else:
            self.quality_weights = quality_weights

        # Initialize dice probabilities
        self.dice_probabilities = self._compute_dice_probabilities()

        # Create board layout (19 tiles in 3-4-5-4-3 pattern)
        self.tiles = self._create_board_layout()

        if tile_config is not None:
            # Use provided configuration
            self._apply_tile_config(tile_config)
        else:
            # Assign resources randomly
            self._assign_resources()
            # Assign number tokens randomly
            self._assign_number_tokens()

        # Build vertex and adjacency structures using exact mappings
        self.vertices = list(range(54))  # 0-53
        self.tiles_touching = self._build_tiles_touching()
        self.vertex_neighbors = self._build_vertex_neighbors()

        # Precompute quality matrices
        self.single_quality = self._precompute_single_quality()
        self.pair_quality = self._precompute_pair_quality()

    @classmethod
    def from_config(cls, tile_config: List[Dict], num_players: int = 4,
                    quality_weights: Optional[Dict[str, float]] = None) -> "Board":
        """
        Create a Board from a specific tile configuration.

        Args:
            tile_config: List of 19 dicts with 'resource' and 'number' keys.
            num_players: Number of players (default: 4)
            quality_weights: Quality function weights (default: equal 1/3 each)

        Returns:
            Board instance with the given configuration
        """
        return cls(num_players=num_players, quality_weights=quality_weights,
                   tile_config=tile_config)
    
    def _compute_dice_probabilities(self) -> Dict[int, float]:
        """Compute probability of rolling each number (2-12)."""
        probs = {}
        for i in range(2, 13):
            # Number of ways to roll this number with two dice
            if i <= 7:
                ways = i - 1
            else:
                ways = 13 - i
            probs[i] = ways / 36.0
        return probs
    
    def _create_board_layout(self) -> List[Dict]:
        """
        Create the standard Catan board layout with 19 hexagonal tiles.
        
        Layout: 3-4-5-4-3 tiles per row
        Tiles are numbered 0-18
        
        Returns:
            List of tile dictionaries with position information
        """
        tiles = []
        
        # Tile positions in rows (using the exact structure)
        # Row 0 (top): 3 tiles (0, 1, 2)
        # Row 1: 4 tiles (3, 4, 5, 6)
        # Row 2 (middle): 5 tiles (7, 8, 9, 10, 11)
        # Row 3: 4 tiles (12, 13, 14, 15)
        # Row 4 (bottom): 3 tiles (16, 17, 18)
        
        tile_rows = [
            (0, 0), (0, 1), (0, 2),  # Row 0: tiles 0-2
            (1, 0), (1, 1), (1, 2), (1, 3),  # Row 1: tiles 3-6
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),  # Row 2: tiles 7-11
            (3, 0), (3, 1), (3, 2), (3, 3),  # Row 3: tiles 12-15
            (4, 0), (4, 1), (4, 2),  # Row 4: tiles 16-18
        ]
        
        for tile_id, (row, col) in enumerate(tile_rows):
            tiles.append({
                'id': tile_id,
                'row': row,
                'col': col,
                'resource': None,  # Will be assigned
                'number': None,     # Will be assigned
            })
        
        return tiles
    
    def _apply_tile_config(self, tile_config: List[Dict]):
        """
        Apply a specific tile configuration to the board.

        Args:
            tile_config: List of 19 dicts, each with 'resource' and 'number' keys.
                        Must have exactly 19 entries (one per tile, in tile order 0-18).

        Raises:
            ValueError: If tile_config is invalid
        """
        valid_resources = set(self.RESOURCE_COUNTS.keys())

        if len(tile_config) != 19:
            raise ValueError(f"tile_config must have exactly 19 entries, got {len(tile_config)}")

        for i, entry in enumerate(tile_config):
            resource = entry.get('resource')
            number = entry.get('number')

            if resource not in valid_resources:
                raise ValueError(
                    f"Tile {i}: invalid resource '{resource}'. "
                    f"Must be one of: {sorted(valid_resources)}"
                )

            if resource == 'desert':
                if number is not None:
                    raise ValueError(f"Tile {i}: desert tiles must have number=None, got {number}")
            else:
                valid_numbers = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}
                if number not in valid_numbers:
                    raise ValueError(
                        f"Tile {i}: invalid number token {number}. "
                        f"Must be one of {sorted(valid_numbers)} (7 is not used in Catan)."
                    )

            self.tiles[i]['resource'] = resource
            self.tiles[i]['number'] = number

    def _assign_resources(self):
        """Randomly assign resources to tiles."""
        resources = []
        for resource, count in self.RESOURCE_COUNTS.items():
            resources.extend([resource] * count)
        
        random.shuffle(resources)
        
        for i, tile in enumerate(self.tiles):
            tile['resource'] = resources[i]
    
    def _assign_number_tokens(self):
        """Randomly assign number tokens to resource tiles (not desert)."""
        # Get all non-desert tiles
        resource_tiles = [t for t in self.tiles if t['resource'] != 'desert']
        
        # Shuffle number tokens
        tokens = self.NUMBER_TOKENS.copy()
        random.shuffle(tokens)
        
        # Assign tokens to resource tiles
        for i, tile in enumerate(resource_tiles):
            tile['number'] = tokens[i]
        
        # Desert has no number token
        for tile in self.tiles:
            if tile['resource'] == 'desert':
                tile['number'] = None
    
    def _build_tiles_touching(self) -> Dict[int, List[int]]:
        """
        Build mapping from vertex to list of tile indices that touch it.
        Uses the exact mapping provided.
        
        Returns:
            Dictionary mapping vertex_id -> list of tile indices
        """
        return {v: tiles.copy() for v, tiles in self.VERTEX_TO_TILES.items()}
    
    def _build_vertex_neighbors(self) -> Dict[int, Set[int]]:
        """
        Build mapping from vertex to set of neighboring vertices (for distance rule).
        Uses the exact mapping provided.
        
        Returns:
            Dictionary mapping vertex_id -> set of neighboring vertex IDs
        """
        return {v: set(neighbors) for v, neighbors in self.VERTEX_NEIGHBORS.items()}
    
    def _precompute_single_quality(self) -> Dict[int, float]:
        """
        Precompute quality score for single settlement at each vertex.
        
        Returns:
            Dictionary mapping vertex_id -> quality score
        """
        single_quality = {}
        
        for vertex in self.vertices:
            vertices_list = [vertex]
            quality = compute_quality(
                vertices_list, self,
                w_resources=self.quality_weights['w_resources'],
                w_expected_cards=self.quality_weights['w_expected_cards'],
                w_prob_at_least_one=self.quality_weights['w_prob_at_least_one']
            )
            single_quality[vertex] = quality
        
        return single_quality
    
    def _precompute_pair_quality(self) -> Dict[int, Dict[int, Dict[int, float]]]:
        """
        Precompute quality score for each player with settlements at v1 and v2.
        
        Returns:
            Dictionary mapping player -> v1 -> v2 -> quality score
        """
        # For now, all players have the same quality function
        # (in the future, we could have player-specific preferences)
        pair_quality = {}
        
        for player in range(1, self.num_players + 1):
            pair_quality[player] = {}
            for v1 in self.vertices:
                pair_quality[player][v1] = {}
                for v2 in self.vertices:
                    if v1 == v2:
                        pair_quality[player][v1][v2] = -float('inf')
                    else:
                        vertices_list = [v1, v2]
                        quality = compute_quality(
                            vertices_list, self,
                            w_resources=self.quality_weights['w_resources'],
                            w_expected_cards=self.quality_weights['w_expected_cards'],
                            w_prob_at_least_one=self.quality_weights['w_prob_at_least_one']
                        )
                        pair_quality[player][v1][v2] = quality
        
        return pair_quality
