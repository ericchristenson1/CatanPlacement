# Catan Optimal Settlement Placement Solver

A solver that finds the optimal initial settlement placement for Catan using depth-first search (DFS) with backward induction, pruning, and memoization.

## Overview

This solver implements a sequential game solver for the initial settlement placement phase of Catan with 2-4 players (configurable). Each player places two settlements in snake order (1→2→3→4→4→3→2→1), and the solver finds the optimal placements assuming all players are fully rational and maximize their own two-settlement quality.

## Features

- **Standard Catan Board**: 19 hexagonal tiles with proper resource and number token distribution
- **Quality Function**: Combines resource diversity, expected cards per turn, and probability of getting at least one resource
- **Optimization Techniques**:
  - Feasibility pruning (Catan distance rules)
  - Local lower bound pruning
  - Upper bound estimation
  - Memoization to avoid recomputing identical subgames
- **Backward Induction**: Uses DFS to solve the sequential game optimally

## Installation

No external dependencies required. Uses only Python 3.8+ standard library.

## Usage

### Basic Solver

Run the solver:

```bash
python main.py [seed] [options]
```

**Arguments:**
- `seed`: Optional random seed for reproducibility
- `--save=path.png` or `-s=path.png`: Save visualization to file
- `--players=N` or `-p=N`: Number of players (2-4, default: 4)
- `--weights=w1,w2,w3` or `-w=w1,w2,w3`: Objetive function weights
  - `w1`: Weight for resource score (default: 1/3)
  - `w2`: Weight for expected cards (default: 1/3)
  - `w3`: Weight for probability at least one (default: 1/3)
  - Weights are automatically normalized to sum to 1

**Examples:**

```bash
# Default (4 players, equal weights)
python main.py

# 3 players with custom seed
python main.py 42 --players=3

# Custom quality weights (50% resources, 30% expected cards, 20% probability)
python main.py --weights=0.5,0.3,0.2

# 3 players with custom weights and save visualization
python main.py 42 --players=3 --weights=0.5,0.3,0.2 --save=board.png

```

### Experimentation

Run experiments comparing different pruning modalities:

```bash
python experiment.py [num_boards] [time_limit] [options]
```

**Arguments:**
- `num_boards`: Number of boards to test (default: 10)
- `time_limit`: Maximum time per execution in seconds (default: 25.0)
- `--modalities=X,Y`: Select specific modalities to test (0, 1, or 2)
  - `0`: Feasibility Pruning Only
  - `1`: Feasibility + Memo
  - `2`: All Prunings (Feasibility + Upper Bound + Memo)
- `--players=N` or `-p=N`: Number of players (2-4, default: 4)
- `--weights=w1,w2,w3` or `-w=w1,w2,w3`: Quality function weights

**Examples:**

```bash
# Default: 10 boards, 25s timeout, all 3 modalities, 4 players
python experiment.py

# Compare only memo vs all prunings with 1 board and large timeout
python experiment.py 1 100 --modalities=1,2

# 3 players with custom weights
python experiment.py 10 25 --players=3 --weights=0.5,0.3,0.2

# Specific modalities in custom order (runs 2 first, then 1)
python experiment.py 1 100 --modalities=2,1 --players=3
```

The experiment compares:
1. Feasibility Pruning Only
2. Feasibility + Memo
3. All Prunings (Feasibility + Upper Bound + Memo)

All modalities are tested on the same set of boards for fair comparison.

Example output:

```
Initializing Catan board...
Board created with 19 tiles and 54 vertices
Resources: ['wood', 'brick', 'wheat', 'ore', 'sheep', 'desert', ...]

Creating solver (with pruning)...

Solving for optimal settlement placements (WITH PRUNING)...
This may take a while depending on board size and pruning effectiveness...

============================================================
SOLUTION FOUND (WITH PRUNING)
============================================================

Player 1 optimal positions: (12, 34)
Player 1 quality score: 8.5234

All players' settlements:
  Player 1: vertices [12, 34], quality: 8.5234
  Player 2: vertices [5, 28], quality: 7.8912
  Player 3: vertices [19, 41], quality: 7.6543
  Player 4: vertices [8, 45], quality: 7.4321

============================================================
PERFORMANCE METRICS
============================================================
Pruning enabled: True
Elapsed time: 2.3456 seconds

Recursive calls: 15,234

Pruning statistics:
  Feasibility prunings: 8,456
  Upper bound prunings: 3,421
  Total prunings: 11,877
  Pruning rate: 77.96%

Memoization statistics:
  Memo hits: 2,345
  Memo misses: 12,889
  Memo size: 1,234
  Hit rate: 15.39%
```

## Architecture

### Files

- **`board.py`**: Board representation with exact vertex-to-tile and vertex-to-neighbor mappings
- **`state.py`**: Game state management during DFS, including settlement placements and feasibility checking
- **`solver.py`**: DFS algorithm with pruning and memoization
- **`quality.py`**: Quality function computation (resource score, expected cards, probability)
- **`main.py`**: Entry point and example usage
- **`visualization_gui.py`**: Graphical visualization using matplotlib
- **`visualization.py`**: Console-based visualization
- **`experiment.py`**: Experimentation script to compare different pruning modalities

### Key Components

#### Board (`board.py`)
- Creates standard Catan board layout (19 tiles in 3-4-5-4-3 arrangement)
- Uses exact vertex-to-tile mappings (54 vertices, each touching 1-3 tiles)
- Uses exact vertex neighbor mappings for distance rule enforcement
- Randomly assigns resources and number tokens
- **Precomputes all quality matrices** (critical for performance):
  - `single_quality[v]`: Quality for each vertex (54 values)
  - `pair_quality[player][v1][v2]`: Quality for every vertex pair (54×54 = 2,916 values total)
  - Currently all players share the same quality function (structure allows future player-specific preferences)
  - This precomputation makes quality evaluation **O(1)** during search instead of O(tiles touched)

The board uses hardcoded mappings:
- `VERTEX_TO_TILES`: Exact mapping of each vertex (0-53) to tiles it touches
- `VERTEX_NEIGHBORS`: Exact mapping of each vertex to its neighbors (for distance rule)

#### State (`state.py`)
- Tracks which players have placed settlements at which vertices
- Implements Catan rules (distance rule, occupancy)
- Provides feasibility checking and upper bound computation
- Generates canonical keys for memoization

#### Solver (`solver.py`)
- Implements DFS with backward induction
- Uses three pruning mechanisms:
  1. **Feasibility pruning**: Skip moves violating Catan rules (distance, occupancy)
  2. **Local LB pruning**: Skip branches with UB ≤ current best value
  3. **Upper bound**: Estimate maximum achievable quality
- **Branch ordering**: Candidates sorted by upper bound/quality to improve LB faster
- Memoizes results to avoid recomputing identical subgames
- Tracks detailed metrics: recursive calls, prunings by type, memo hits/misses

#### Quality Function (`quality.py`)
The quality function combines three components:

1. **Resource Score**: Measures resource type diversity and coverage
2. **Expected Cards**: Expected number of cards per turn based on dice probabilities
3. **Probability at Least One**: Probability of getting at least one resource in a turn

Default weights: `w_resources = 1/3`, `w_expected_cards = 1/3`, `w_prob_at_least_one = 1/3`

**Configurable weights**: You can customize the quality function weights via command-line arguments:
- In `main.py`: `--weights=w1,w2,w3`
- In `experiment.py`: `--weights=w1,w2,w3`

Weights are automatically normalized to sum to 1, so `--weights=0.5,0.3,0.2` is equivalent to `--weights=5,3,2`.

## Algorithm

The solver uses a recursive DFS that implements the snake order placement:

1. Player 1 places first settlement
2. Recursively, players 2, 3, ... (up to N players) place their first settlements
3. After recursion unwinds, each player places their second settlement optimally
4. The recursion structure naturally produces the order: 1→2→...→N→N→...→2→1

The number of players is configurable (2-4), allowing you to test different game scenarios.

At each node:
- Get all feasible first positions
- **Sort candidates by upper bound/quality** (best first) to establish strong LB early
- For each position:
  - **Compute upper bound BEFORE branching**: For the candidate first position, calculate the maximum quality achievable if the player could place both settlements optimally without other players interfering (relaxation). This is done by finding the maximum `pair_quality(first_pos, second_pos)` over all currently feasible second positions.
  - **Prune if UB ≤ LB**: If the upper bound is less than or equal to the best value (LB) found so far at this node, prune the entire branch without exploring it recursively. This is safe because even in the optimistic scenario (no interference), this branch cannot improve the solution.
  - If not pruned: Place first settlement and recurse on later players
  - After recursion unwinds, choose best second position using precomputed `pair_quality`
- Update local best value (LB) as better solutions are found
- Strong LB (from exploring good candidates first) enables more pruning in subsequent iterations

## Performance Metrics

The solver tracks detailed performance metrics:

- **Recursive calls**: Total number of DFS recursive calls
- **Pruning statistics**:
  - Feasibility prunings: Branches pruned due to Catan rules
  - Upper bound prunings: Branches pruned due to upper bound ≤ local LB
  - Total prunings and pruning rate
- **Memoization statistics**:
  - Memo hits/misses and hit rate
  - Cache size
- **Execution time**: Elapsed time in seconds

### Comparing Pruning Modalities

Use the experiment script to compare different pruning strategies:

```bash
python experiment.py 10 25
```

This will:
- Generate 10 different boards
- Test each board with 3 different pruning modalities:
  1. Solo Feasibility Pruning
  2. Feasibility + Memo
  3. All Prunings (Feasibility + Upper Bound + Memo)
- Show average times, min/max times, and timeout counts for each modality
- Calculate speedup factors

The experiment uses threading to enforce a real timeout (default 25 seconds), cutting off executions that exceed the limit.

You can also compare with and without pruning in the main solver:

```bash
python main.py 42 --compare
```

## Performance Optimization

The solver uses several aggressive optimization techniques to handle the large search space efficiently:

### Precomputation (O(1) Evaluation)

**All quality scores are precomputed before search begins:**
- `single_quality[v]`: Quality score for each vertex (54 precomputed values)
- `pair_quality[player][v1][v2]`: Quality score for every pair of vertices (54×54 = 2,916 precomputed values)

**Note**: Currently all players use the same quality function, so `pair_quality[1][v1][v2] == pair_quality[2][v1][v2] == ...` for all players. The structure is indexed by player to allow future extensions where different players might have different preferences (e.g., different resource weights, different strategies).

This means evaluating a settlement placement is **O(1)** - just a dictionary lookup. Without precomputation, each evaluation would require recalculating resource scores, expected cards, and probabilities, which would be orders of magnitude slower.

### Branch Ordering (Strong Lower Bound Early)

**Candidates are sorted by quality/upper bound before exploration:**
- When exploring first settlement positions, candidates are sorted in descending order by their upper bound (or single quality if upper bound pruning is disabled)
- This ensures we explore the **best individual options first**
- Finding a strong solution early gives us a **strong lower bound (LB)** quickly
- A strong LB enables more aggressive pruning: branches with UB ≤ LB are immediately discarded
- This creates a positive feedback loop: better LB → more pruning → faster search

**Impact**: This technique alone can reduce search time by 2-5x by finding good solutions early and pruning more branches.

### Pruning Techniques

1. **Feasibility Pruning**: Skips moves violating Catan rules (distance, occupancy) - eliminates invalid branches immediately

2. **Upper Bound (UB) Pruning**: 
   - **Calculation**: Before exploring a branch (before recursion), for each candidate first settlement position, the solver computes an upper bound on the maximum quality the player could achieve.
   - **Intuition**: The UB is a **relaxation** that assumes the player could place both settlements optimally **without other players restricting their choices**. Specifically:
     - Given a first settlement at position `v1`, the UB is the maximum `pair_quality(v1, v2)` over all currently feasible positions `v2` for the second settlement
     - This ignores that future players may take some of these positions, making it an optimistic (upper) bound
   - **Pruning logic**: If `UB ≤ LB` (where LB is the best value found so far for this player at this node), the entire branch is pruned without exploration, because even in the best-case scenario (no other players interfering), this branch cannot beat the current best solution
   - **Effectiveness**: This is particularly powerful when combined with branch ordering, as exploring good candidates first establishes a strong LB early, enabling more aggressive pruning of remaining branches

3. **Local Lower Bound (LB)**: Maintained per recursive call, updated as better solutions are found. Enables pruning relative to the best known value at that node

### Memoization

- Caches results for identical game states (same player, same occupied vertices, same available vertices)
- Avoids recomputing entire subtrees when the same configuration is reached via different paths
- Particularly effective in later stages of the game when fewer options remain

### Combined Impact

These techniques work synergistically:
- **Precomputation** makes evaluation instant (O(1))
- **Branch ordering** finds strong LB early
- **Strong LB** enables more **upper bound pruning**
- **Memoization** avoids redundant computation
- **Feasibility pruning** eliminates invalid moves upfront

**Typical speedup**: 10-100x faster than feasibility-only pruning, and 100-1000x faster than naive DFS without optimizations.

The actual runtime depends on:
- Board size (number of vertices)
- Effectiveness of pruning (varies with board difficulty)
- Number of feasible placements at each step
- Board difficulty (resource/number distribution affects quality variance)

## Visualization

The solver includes both console and graphical visualization:

- **Console visualization** (`visualization.py`): Text-based output showing board layout and settlements
- **Graphical visualization** (`visualization_gui.py`): Matplotlib-based visualization showing:
  - Hexagonal tiles with resource colors and numbers
  - Player settlements as colored circles at vertices
  - Legend with resource types and player colors

Visualization is automatically shown after solving, or can be saved with `--save=filename.png`

## Implementation Details

### Board Structure

The board uses exact mappings for a standard Catan layout:
- **19 tiles** numbered 0-18 in a 3-4-5-4-3 row arrangement
- **54 vertices** numbered 0-53
- Each vertex touches 1, 2, or 3 tiles (never 4)
- Exact neighbor relationships for distance rule enforcement

### Quality Function

The quality function combines three components with equal weights (1/3 each):
1. **Resource Score**: Measures resource type diversity and coverage
2. **Expected Cards**: Expected number of cards per turn based on dice probabilities
3. **Probability at Least One**: Probability of getting at least one resource in a turn

### Solver Algorithm

The solver uses:
- **DFS with backward induction**: Implements snake order (1→2→3→4→4→3→2→1)
- **Local lower bound**: Maintained per recursive call, updated as better solutions are found
- **Upper bound estimation**: For each candidate first settlement, estimates maximum achievable quality
- **Branch ordering**: Explores best candidates first to improve LB faster
- **Memoization**: Caches results for identical game states

## Configuration

### Number of Players

The solver supports 2-4 players. Configure via:
- `main.py`: `--players=N` or `-p=N`
- `experiment.py`: `--players=N` or `-p=N`

### Quality Function Weights

Customize the quality function weights to emphasize different aspects:
- **Higher resource weight**: Prioritizes resource diversity
- **Higher expected cards weight**: Prioritizes high-probability number tokens
- **Higher probability weight**: Prioritizes reliability (getting at least one card per turn)

Configure via:
- `main.py`: `--weights=w1,w2,w3` or `-w=w1,w2,w3`
- `experiment.py`: `--weights=w1,w2,w3` or `-w=w1,w2,w3`

## Limitations

- All players use the same quality function (could be extended for player-specific preferences)
- Timeout in experiments uses threading (may not work perfectly in all environments)

## Repository

This project is available on GitHub:
- Repository: https://github.com/ericchristenson1/CatanPlacement

## Future Improvements

- Player-specific quality functions
- Parallel search for faster execution
- More sophisticated timeout mechanisms
- Export experiment results to CSV/JSON
- Support for more than 4 players

## License

This is an educational implementation for solving Catan settlement placement optimization.

