"""
Custom board input framework for the Catan settlement placement solver.

Allows you to specify an exact Catan board (resources + numbers) and then
runs the optimal settlement placement solver on it.

Usage:
    python solve_custom.py                          # Interactive input
    python solve_custom.py --load=board.json        # Load from JSON file
    python solve_custom.py --load=board.json --players=3
    python solve_custom.py --load=board.json --save=result.png
    python solve_custom.py --load=board.json --save-config=saved.json
    python solve_custom.py --load-csv=board.csv     # Load from CSV file
    python solve_custom.py --lc=board.csv --players=3

JSON format (list of 19 tiles in order 0-18):
    [
        {"resource": "wood",   "number": 11},
        {"resource": "sheep",  "number": 12},
        ...
        {"resource": "desert", "number": null}
    ]

CSV format (19 data rows, optional header, row order = tile 0-18):
    resource,number
    wood,11
    sheep,12
    wheat,9
    ...
    desert,

    - Column 1: resource name or alias (w, b, wh, o, s, d)
    - Column 2: number 2-6 or 8-12; leave blank for desert
    - Header row auto-detected and skipped if present

Tile layout (IDs 0-18):
         [ 0] [ 1] [ 2]
       [ 3] [ 4] [ 5] [ 6]
     [ 7] [ 8] [ 9] [10] [11]
       [12] [13] [14] [15]
         [16] [17] [18]
"""

import sys
import json
from board import Board
from solver import Solver
from visualization_gui import visualize_board_gui

VALID_RESOURCES = ['wood', 'brick', 'wheat', 'ore', 'sheep', 'desert']
RESOURCE_ALIASES = {
    'w': 'wood', 'wo': 'wood',
    'b': 'brick', 'br': 'brick',
    'wh': 'wheat', 'grain': 'wheat', 'g': 'wheat',
    'o': 'ore',
    's': 'sheep', 'sh': 'sheep',
    'd': 'desert', 'de': 'desert',
}

# Standard Catan board layout for display
BOARD_DIAGRAM = """
Tile layout (enter data for each tile ID):

           [ 0] [ 1] [ 2]
         [ 3] [ 4] [ 5] [ 6]
       [ 7] [ 8] [ 9] [10] [11]
         [12] [13] [14] [15]
           [16] [17] [18]

Resources: wood (w), brick (b), wheat (wh), ore (o), sheep (s), desert (d)
Numbers:   2-6, 8-12  (7 is not used; no number for desert)
"""

TILE_ROW_LABELS = [
    "Row 0 (top)   : tiles  0,  1,  2",
    "Row 1         : tiles  3,  4,  5,  6",
    "Row 2 (middle): tiles  7,  8,  9, 10, 11",
    "Row 3         : tiles 12, 13, 14, 15",
    "Row 4 (bottom): tiles 16, 17, 18",
]


def parse_resource(raw: str) -> str:
    """Normalize resource input string to a valid resource name."""
    raw = raw.strip().lower()
    if raw in VALID_RESOURCES:
        return raw
    if raw in RESOURCE_ALIASES:
        return RESOURCE_ALIASES[raw]
    return None


def input_tile_interactive(tile_id: int) -> dict:
    """Interactively prompt user for resource and number for a single tile."""
    while True:
        raw = input(f"  Tile {tile_id:2d} - resource: ").strip()
        resource = parse_resource(raw)
        if resource is None:
            print(f"    Invalid resource '{raw}'. Use: {', '.join(VALID_RESOURCES)}")
            continue
        break

    if resource == 'desert':
        return {'resource': 'desert', 'number': None}

    valid_numbers = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}
    while True:
        raw = input(f"  Tile {tile_id:2d} - number token (2-6, 8-12): ").strip()
        try:
            number = int(raw)
            if number in valid_numbers:
                break
            elif number == 7:
                print("    7 is not a valid number token in Catan (7 triggers the robber).")
            else:
                print(f"    Number must be 2-6 or 8-12, got {number}.")
        except ValueError:
            print(f"    Invalid number '{raw}'. Enter an integer (2-6 or 8-12).")

    return {'resource': resource, 'number': number}


def input_board_interactive() -> list:
    """Walk the user through entering all 19 tiles interactively."""
    print(BOARD_DIAGRAM)
    print("Enter resource and number for each tile.\n")

    tile_config = []
    row_breaks = [0, 3, 7, 12, 16, 19]  # tile IDs where each row starts

    for row_idx, label in enumerate(TILE_ROW_LABELS):
        start = row_breaks[row_idx]
        end = row_breaks[row_idx + 1]
        print(f"{label}")
        for tile_id in range(start, end):
            entry = input_tile_interactive(tile_id)
            tile_config.append(entry)
        print()

    return tile_config


def load_board_from_json(path: str) -> list:
    """Load tile configuration from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) != 19:
        raise ValueError(f"JSON file must contain a list of exactly 19 tile entries, got {len(data)}")

    tile_config = []
    for i, entry in enumerate(data):
        resource = entry.get('resource', '').lower()
        number = entry.get('number')
        if resource not in VALID_RESOURCES:
            raise ValueError(f"Tile {i}: invalid resource '{resource}'")
        valid_numbers = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}
        if resource != 'desert' and (number is None or int(number) not in valid_numbers):
            raise ValueError(
                f"Tile {i}: number must be 2-6 or 8-12 for non-desert tiles, got {number} "
                f"(7 is not a valid number token in Catan)"
            )
        tile_config.append({
            'resource': resource,
            'number': int(number) if number is not None else None
        })

    return tile_config


def load_board_from_csv(path: str) -> list:
    """Load tile configuration from a CSV file."""
    import csv
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Auto-skip header row if first cell isn't a valid resource
    if rows and parse_resource(rows[0][0]) is None:
        rows = rows[1:]

    if len(rows) != 19:
        raise ValueError(f"CSV must have exactly 19 data rows, got {len(rows)}")

    valid_numbers = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}
    tile_config = []
    for i, row in enumerate(rows):
        if len(row) < 2:
            raise ValueError(f"Row {i}: expected 2 columns (resource, number), got {len(row)}")
        resource = parse_resource(row[0].strip())
        if resource is None:
            raise ValueError(f"Tile {i}: invalid resource '{row[0].strip()}'")
        raw_num = row[1].strip().lower()
        if resource == 'desert':
            number = None
        else:
            if raw_num in ('', 'null', 'none'):
                raise ValueError(f"Tile {i}: non-desert tile must have a number")
            try:
                number = int(raw_num)
            except ValueError:
                raise ValueError(f"Tile {i}: invalid number '{row[1].strip()}'")
            if number not in valid_numbers:
                raise ValueError(
                    f"Tile {i}: number must be 2-6 or 8-12, got {number} "
                    f"(7 is not a valid number token in Catan)"
                )
        tile_config.append({'resource': resource, 'number': number})

    return tile_config


def save_config_to_json(tile_config: list, path: str):
    """Save tile configuration to a JSON file for reuse."""
    with open(path, 'w') as f:
        json.dump(tile_config, f, indent=2)
    print(f"Board configuration saved to: {path}")


def print_board_summary(tile_config: list):
    """Print a summary of the board configuration."""
    print("Board configuration:")
    rows = [(0, 3), (3, 7), (7, 12), (12, 16), (16, 19)]
    row_indent = ["      ", "    ", "  ", "    ", "      "]
    for idx, (start, end) in enumerate(rows):
        line = row_indent[idx]
        for tile_id in range(start, end):
            t = tile_config[tile_id]
            res = t['resource'][:2].upper()
            num = str(t['number']) if t['number'] is not None else '--'
            line += f"[{tile_id:02d}:{res}/{num:>2}] "
        print(line)
    print()


def main():
    load_path = None
    load_csv_path = None
    save_image = None
    save_config_path = None
    num_players = 4
    quality_weights = None

    for arg in sys.argv[1:]:
        if arg.startswith("--load=") or arg.startswith("-l="):
            load_path = arg.split("=", 1)[1]
        elif arg.startswith("--load-csv=") or arg.startswith("--lc="):
            load_csv_path = arg.split("=", 1)[1]
        elif arg.startswith("--save=") or arg.startswith("-s="):
            save_image = arg.split("=", 1)[1]
        elif arg.startswith("--save-config=") or arg.startswith("--sc="):
            save_config_path = arg.split("=", 1)[1]
        elif arg.startswith("--players=") or arg.startswith("-p="):
            try:
                num_players = int(arg.split("=", 1)[1])
                if not (2 <= num_players <= 4):
                    print(f"Error: players must be 2-4, got {num_players}")
                    return
            except ValueError:
                print(f"Error: invalid --players value: {arg}")
                return
        elif arg.startswith("--weights=") or arg.startswith("-w="):
            try:
                parts = [float(x) for x in arg.split("=", 1)[1].split(",")]
                if len(parts) != 3:
                    print("Error: --weights requires exactly 3 values (w1,w2,w3)")
                    return
                total = sum(parts)
                if total <= 0:
                    print("Error: weights must sum to a positive number")
                    return
                quality_weights = {
                    'w_resources': parts[0] / total,
                    'w_expected_cards': parts[1] / total,
                    'w_prob_at_least_one': parts[2] / total,
                }
            except ValueError:
                print(f"Error: invalid --weights value: {arg}")
                return
        elif arg in ("--help", "-h"):
            print(__doc__)
            return
        else:
            print(f"Unknown argument: {arg}. Use --help for usage.")
            return

    # --- Get tile configuration ---
    if load_path is not None:
        print(f"Loading board from: {load_path}")
        try:
            tile_config = load_board_from_json(load_path)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"Error loading board: {e}")
            return
    elif load_csv_path is not None:
        print(f"Loading board from CSV: {load_csv_path}")
        try:
            tile_config = load_board_from_csv(load_csv_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading CSV: {e}")
            return
    else:
        tile_config = input_board_interactive()

    # Optionally save config
    if save_config_path is not None:
        save_config_to_json(tile_config, save_config_path)

    # Print board summary
    print_board_summary(tile_config)

    # --- Build board ---
    print(f"Initializing board ({num_players} players)...")
    try:
        board = Board.from_config(tile_config, num_players=num_players,
                                  quality_weights=quality_weights)
    except ValueError as e:
        print(f"Error building board: {e}")
        return

    print(f"Board created: {len(board.tiles)} tiles, {len(board.vertices)} vertices")
    print()

    # --- Solve ---
    print("Solving for optimal settlement placements (with full pruning)...")
    print("This may take a moment...\n")

    solver = Solver(board, enable_pruning=True)
    final_state, player1_positions, player1_quality = solver.solve()

    if final_state is None:
        print("ERROR: No valid solution found.")
        return

    # --- Print results ---
    print("=" * 60)
    print("OPTIMAL SETTLEMENT PLACEMENTS")
    print("=" * 60)
    print()

    for player in range(1, num_players + 1):
        houses = final_state.houses[player]
        if len(houses) == 2:
            quality = final_state.quality_of_player(player)
            # Show what resources/numbers each vertex gives
            v1, v2 = houses
            print(f"  Player {player}: vertices {houses}  (objective: {quality:.4f})")
        else:
            print(f"  Player {player}: vertices {houses} (incomplete)")

    print()
    solver.print_metrics()

    # --- Visualize ---
    print("Generating visualization...")
    try:
        visualize_board_gui(board, final_state, save_path=save_image)
    except ImportError:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating visualization: {e}")


if __name__ == "__main__":
    main()
