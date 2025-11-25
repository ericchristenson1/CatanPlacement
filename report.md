**15.C57 Optimization**  
**Christenson, Lobon, Martino**

---

## Optimal Initial Settlement Placement in Catan: A Game-Theoretic Approach

### Problem Statement
The goal of this project is to determine the optimal placement of the two initial settlements in Catan assuming that all players act optimally and symmetrically.
 In Catan, players place settlements in the order 1–2–3–4–4–3–2–1, meaning early moves constrain future options. The first player’s choice shapes the entire game, since each subsequent placement reduces the feasible set for everyone else, including their own second move.
 This sequential dependency naturally forms a game tree, where each node represents a partial configuration of placed settlements and each branch corresponds to a response by the next player. Our goal is to identify an optimal root-to-leaf path that preserves optimal play for every player at each decision point. However, explicitly constructing the entire game tree quickly becomes computationally ineficient due to its exponential growth in the number of possible placements.
We will focus on developing computational techniques to efficiently solve the game tree of optimal placements under the stated assumptions.
In this simplified version, we will not consider roads, ports, monopoly strategies, or adversarial positioning aimed at harming other players. These additional strategic elements could be incorporated as future extensions to the model.


#### Quick Primer on Catan Setup
- **Board**: a fixed arrangement of 19 hex tiles (three rows of 3-4-5-4-3) representing the resources brick, lumber, ore, grain, and wool plus a desert. Each hex is labeled with a dice number from 2 to 12 (excluding 7) and produces its resource when that number is rolled.
- **Vertices (intersections)**: settlements can be placed on the 54 intersections formed by the hex grid. Each vertex touches up to three hexes; resource production for a settlement equals the sum of contributions from all adjacent hexes.
- **Dice probabilities**: probability of each number is `Prob(k) = combinations(k) / 36`, e.g., `Prob(6) = Prob(8) = 5/36`, `Prob(4) = Prob(10) = 3/36`, etc. These probabilities translate into expected resource inflows.
- **Distance rule (feasibility)**: two settlements cannot be adjacent; at least two edges must separate any pair. Additionally, a settlement must touch at least one hex that still has resource supply (the desert provides none).

We deliberately ignore roads, ports, robber movement, and development cards to focus on the pure placement stage; they can be layered on later once the solver core is validated.

Simplifying assumptions purposely exclude roads, ports, monopolies, robber control, or spite-driven blocking. These elements can appear as future layers once the core solver is performant.

### Player objective function
We define the following composite score that every player is assumed to maximize optimally and symmetrically. Given the two vertices `S_p = {s₁, s₂}` owned by player `p`, we compute three interpretable metrics and combine them with designer-selected weights `(α, β, γ)`:
  1. **Material diversity**: counts how many distinct resource types the two settlements can produce (maximum of five). This rewards coverage across brick, lumber, ore, grain, and wool.
  2. **Expected cards per turn**: sums all adjacent dice probabilities to estimate the expected number of resource cards the player receives on each roll.
  3. **Chance of receiving at least one card**: aggregates the unique dice probabilities (ignoring repeated numbers) to capture how often any of the player’s numbers are likely to hit, favoring a spread of dice values.
  
  The objective used by the solver is `Objective_p = α·Prod(S_p) + β·Bal(S_p) + γ·Scar(S_p)`. 

### Backward Induction
This is a deterministic dynamic-programming problem that we solve via backward induction. Because the tree is shallow but extremely wide, we traverse it with a depth-first search

**pseudocode**:

```
function Solve(board, player):
    if player > num_players:
        return board
    
    best_board = None
    best_objective = -∞
    
    for vertex in feasible_vertices(board):
        new_board = Clone(board)
        PlaceSettlement(new_board, player, vertex)
        
        new_board = Solve(new_board, player + 1)
        
        // Now place this player's second settlement optimally
        // (given their first settlement and all later players' placements)
        PlaceSettlement(new_board, player, best_second_vertex(vertex,board))
        
        objective_value = Objective(new_board, player)
        
        if objective_value > best_objective:
            best_objective = objective_value
            best_board = new_board
    
    return best_board
```

The function `Solve(board, player)` implements backward induction by recursively solving from the perspective of player `player` who is about to place their first settlement. The key insight is that each player's optimal decision depends on how later players will respond optimally.

For each feasible first position, the algorithm: (1) clones the board and places the first settlement, (2) recursively calls `Solve` for the next player, which eventually solves for all remaining players, and (3) after the recursion returns, places this player's second settlement optimally given the constraints imposed by all later players' optimal placements.

The recursion naturally implements the snake order `1→2→3→4→4→3→2→1`. 

### Memoization
Many game states share identical configurations of occupied vertices and available feasible options, even if reached through different placement sequences. For each player `p` and a given set of feasible vertices (available options), the optimal resulting board configuration is the same regardless of how we arrived at that state. By caching the mapping `(player, occupied_vertices, available_vertices) → best_resulting_board`, we avoid recomputing entire subtrees when the same configuration is encountered again.

The memoization key combines the current player and the set of available vertices. This allows us to recognize equivalent states even when reached via different paths through the game tree. When a memo hit occurs, we can immediately return the cached optimal board configuration without any recursive exploration.

**pseudocode** (extending the prevous base):

```
memo = {}  // Maps (player, available_vertices) → best_board

function Solve(board, player):
    if player > num_players:
        return board
    
    memo_key = (player, available_vertices)
    
    if memo_key in memo:
        return memo[memo_key]
    
    best_board = None
    best_objective = -∞
    
    for vertex in feasible_vertices(board, player):
        new_board = Clone(board)
        PlaceSettlement(new_board, player, vertex)
        
        new_board = Solve(new_board, player + 1)
        
        if new_board is None:
            continue
        
        PlaceSettlement(new_board, player, best_second_vertex(vertex, new_board))
        
        objective_value = Objective(new_board, player)
        
        if objective_value > best_objective:
            best_objective = objective_value
            best_board = new_board
    
    memo[memo_key] = best_board
    return best_board
```

### Pruning Strategies
- **Lower bound (LB)**: At each recursive node, we maintain a local `best_objective` value representing the best complete solution found so far for the current player at this node. This acts as a lower bound: any branch that cannot exceed this value is pruned. The LB is updated dynamically as we explore branches and discover better solutions.

- **Upper bound (UB)**: Before making a recursive call for a candidate first settlement position, we compute an optimistic upper bound on the maximum objective value achievable from that branch. The UB assumes an optimistic relaxation: given the first settlement at position `v₁`, we calculate the maximum `pair_quality(v₁, v₂)` over all currently feasible second positions `v₂`, ignoring that future players may take some of these positions. If `UB ≤ LB`, we prune the entire branch without recursion, since even in the best-case scenario (no interference from other players), this branch cannot improve upon the current best solution. To make UB computation extremely efficient, we precompute a `pair_quality` matrix of size `n × n` (where `n` is the number of vertices) at initialization, storing the objective function value for every possible vertex pair `(v₁, v₂)`. This precomputation takes O(n²) time once, making each UB query during search a simple lookup: for a given first vertex `v₁`, the UB is just the maximum value in row `v₁` of this precomputed matrix over all currently feasible second vertices.

- **Move ordering**: We sort candidate first positions by their individual vertex quality (single-settlement objective) in descending order before exploration. This ensures we explore the most promising branches first, which helps establish a strong LB early in the search. A strong LB enables more aggressive pruning of remaining branches.

**pseudocode** (extending the prevous base):

```
memo = {}

function Solve(board, player):
    if player > num_players:
        return board
    
    memo_key = (player, GetAvailableVertices(board))
    
    if memo_key in memo:
        return memo[memo_key]
    
    best_board = None
    best_objective = -∞  // Local lower bound        
    
    for vertex in sorted(feasible_vertices(board, player)):

        ub = UpperBound(board, player, vertex)
        if ub <= best_objective:
            continue  // Prune this branch
        
        new_board = Clone(board)
        PlaceSettlement(new_board, player, first_vertex)
        
        new_board = Solve(new_board, player + 1)
        
        if new_board is None:
            continue
        
        PlaceSettlement(new_board, player, BestSecondVertex(new_board, player, first_vertex))
        
        objective_value = Objective(new_board, player)
        
        if objective_value > best_objective:
            best_objective = objective_value
            best_board = new_board
    
    memo[memo_key] = best_board
    return best_board
```

Memoization avoids redundant computation of identical subtrees, while pruning eliminates hopeless branches before expensive recursive exploration. Move ordering amplifies both effects by finding strong solutions early, which tightens LBs and enables more aggressive pruning.

### Experimental Setup
To quantify the performance benefits of each optimization technique, we conducted a comprehensive experimental evaluation. We generated 30 random Catan boards and evaluated each board using three solver modalities:

1. **Feasibility Pruning Only**: Baseline DFS that only filters out infeasible moves (violations of distance rule and occupancy constraints). This serves as our reference baseline.

2. **Feasibility + Memoization**: Adds memoization to cache results for identical game states, avoiding redundant subtree exploration.

3. **All Prunings (Feasibility + Upper Bound + Memoization)**: The complete solver with all optimizations enabled, including upper bound pruning and move ordering.

For each board and modality, we measured:
- **Execution time**: Total wall-clock time to find the optimal solution
- **Recursive calls**: Total number of DFS recursive invocations, which directly reflects the size of the search space explored
- **Solution verification**: We compared the optimal solutions found by all three modalities to ensure correctness—all modalities must agree on the optimal placement, confirming that pruning does not eliminate optimal branches.

This experimental design allows us to isolate and quantify the contribution of each optimization technique, demonstrating both the correctness (all modalities find identical optimal solutions) and the dramatic performance improvements achieved through memoization and pruning.

### Results and Benchmarks
The experimental results on 30 random boards demonstrate the substantial performance gains achieved by each optimization technique:

**Performance Metrics**:

| Modality | Average Time | Min Time | Max Time | Avg Recursive Calls | Speedup |
|----------|--------------|----------|----------|---------------------|---------|
| Feasibility Pruning Only | 444.16s | 441.64s | 454.26s | 5,634,937 | 1.00× (baseline) |
| Feasibility + Memoization | 79.20s | 78.61s | 80.28s | 983,839 | 5.61× |
| All Prunings (Full Solver) | 0.33s | 0.05s | 0.69s | 569 | 1,361.88× |

**Key Findings**:

- **Memoization impact**: Adding memoization reduces execution time by a factor of 5.61× (from 444s to 79s) and recursive calls by 82.5% (from 5.6M to 984K). This demonstrates that many game states are reached through multiple paths, and caching these results eliminates massive redundant computation.

- **Upper bound pruning impact**: The complete solver with all optimizations achieves a **1,361× speedup** over the baseline, reducing average execution time from 444 seconds to just 0.33 seconds. Recursive calls drop by 99.99% (from 5.6M to 569 on average), showing that upper bound pruning eliminates the vast majority of hopeless branches before expensive recursive exploration.

- **Solution correctness**: All three modalities found identical optimal solutions for all 30 boards, confirming that neither memoization nor pruning eliminates optimal branches. This validates the correctness of our optimization techniques.

- **Consistency**: The baseline modality shows remarkably consistent performance across all boards (5,634,937 recursive calls for every board), indicating that the search space size is largely independent of board configuration when only feasibility pruning is used. In contrast, the full solver shows variability (65 to 1,195 recursive calls), reflecting how different board configurations affect the effectiveness of upper bound pruning.

### Possible Extensions

- **Competitiveness-aware objectives**: Instead of maximizing their own score `v_p`, each player could maximize `v_p − max_{q≠p} v_q` (their advantage over the best opponent). This is more realistic because players care about winning, not just absolute value—they're motivated to both maximize their own benefit and limit opponents' advantage. This increases tree depth: player 4's second move now affects how players 1–3 respond in their second placements (since they care about relative performance), creating order `1→2→3→4→4→3→2→1`. The UB/LB pruning extends naturally: `UB_competitive(p) = UB_self(p) − LB_maxOthers`.
- **Stochastic opponents**: Instead of assuming all players act optimally, we can model realistic error behavior for players 2–4 (e.g., using a logit choice model where players are more likely to choose better moves but can still make mistakes). Player 1 then solves a stochastic dynamic program, maximizing expected utility while integrating over the probability distribution of other players' moves. This is more realistic because real players make suboptimal decisions, and player 1's optimal strategy should account for these uncertainties rather than assuming perfect play from opponents.
- **New objective functions**: incorporate monopoly bonuses (“maximize wheat dominance”), port leverage, or robber resilience. These alter `Evaluate(state)` but reuse the solver skeleton.
- **Full game features**: integrate roads, ports, or road-building constraints, expanding the state but still capturable by the memoized DFS with carefully designed feasibility checks.

### Conclusions
- The solver demonstrates how backward induction, memoization, and pruning convert an explosive game tree into a tractable optimization pipeline, quantifiably outperforming naive DFS at each incremental enhancement.
- Because each acceleration is modular (caching layer, UB/LB estimators, move ordering), the framework adapts quickly to richer rulesets—making it a strong foundation for increasingly realistic Catan simulators.
- The methodology generalizes to sequential competitive placement problems where each decision constrains future feasibility (e.g., facility location with exclusivity, draft picks). It showcases the tight integration of game theory, optimization, and algorithmic design to compute exact equilibria efficiently.

---

## Appendix

### A.1 Example Optimal Placement Visualization

The following visualization shows an example of optimal settlement placement computed by the solver. This placement was generated by running `main.py` with the following objective function weights:

- **Resource diversity weight (α)**: 0.500
- **Expected cards per turn weight (β)**: 0.300  
- **Probability of at least one card weight (γ)**: 0.200

These weights emphasize material diversity (50% weight) while still valuing expected resource production (30%) and dice number spread (20%).

![Optimal Settlement Placement](Figure_1.png)



