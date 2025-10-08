# ASCII Neural Network Visualizer
# This cell defines a function `ascii_nn` and prints a demo for layers [2,4,4,1].
from math import floor, ceil
from typing import List, Dict, Union, Tuple

def _parse_layers(spec: Union[List[int], Dict[str, int]]) -> Tuple[List[str], List[int]]:
    if isinstance(spec, dict):
        names = list(spec.keys())
        sizes = [int(spec[k]) for k in names]
    else:
        sizes = [int(x) for x in spec]
        names = [f"layer_{i}" for i in range(len(sizes))]
        names[0] = "input"
        names[-1] = "output"
        if len(sizes) > 2:
            for i in range(1, len(sizes)-1):
                names[i] = f"hidden_{i}"
    return names, sizes

def ascii_nn(
    spec: Union[List[int], Dict[str,int]],
    max_height: int = 20,
    col_spacing: int = 6,
    node_char: str = "o",
    edge_char: str = ".",
    edge_cap: int = 1200,
    show_legend: bool = True
) -> str:
    """
    Render a fully-connected feedforward NN as ASCII.
    - spec: e.g., [2, 4, 4, 1] or {'input':2,'hidden_1':4,'hidden_2':4,'output':1}
    - max_height: maximum rows used to display node columns (scaled if layers exceed this)
    - col_spacing: columns between layer node columns
    - edge_cap: maximum number of edges to draw before uniformly sampling
    """
    names, sizes = _parse_layers(spec)
    L = len(sizes)
    if L < 2:
        return "Need at least two layers."

    # Determine canvas height (rows only for node columns; we'll add label rows later)
    original_max = max(sizes)
    rows = min(original_max, max_height)
    cols = (L - 1) * col_spacing + 1

    # Map each layer's node indices to row positions on the scaled canvas
    def layer_positions(n: int) -> List[int]:
        if n == 1:
            return [rows // 2]
        # Spread nodes evenly across [0, rows-1]
        return [round(i * (rows - 1) / (n - 1)) for i in range(n)]

    layer_rows = [layer_positions(n) for n in sizes]
    layer_cols = [i * col_spacing for i in range(L)]

    # Initialize canvas (add 3 lines for header/legend later)
    pad_top = 2  # room for layer names
    canvas = [[" " for _ in range(cols)] for _ in range(rows + pad_top)]

    # Place layer name labels centered above columns
    for name, c in zip(names, layer_cols):
        start = max(0, c - len(name)//2)
        end = min(cols, start + len(name))
        for j, ch in enumerate(name[:end-start]):
            canvas[0][start + j] = ch

    # Draw nodes
    for c, positions in zip(layer_cols, layer_rows):
        for r in positions:
            canvas[r + pad_top][c] = node_char

    # Determine edges to draw (sampling if too many)
    total_edges = sum(sizes[i] * sizes[i+1] for i in range(L-1))
    sampled = False
    edges_to_draw = []
    if total_edges <= edge_cap:
        for i in range(L-1):
            c1, c2 = layer_cols[i], layer_cols[i+1]
            for r1 in layer_rows[i]:
                for r2 in layer_rows[i+1]:
                    edges_to_draw.append((r1 + pad_top, c1, r2 + pad_top, c2))
    else:
        # Uniformly sample edges across the whole network
        sampled = True
        # target samples per adjacent-layer pair proportional to edges
        for i in range(L-1):
            n1, n2 = sizes[i], sizes[i+1]
            pair_edges = n1 * n2
            k = max(1, round(edge_cap * (pair_edges / total_edges)))
            # Create deterministic samples by stepping through index grid
            step = max(1, round((pair_edges) / k))
            c1, c2 = layer_cols[i], layer_cols[i+1]
            idx = 0
            count = 0
            for a in range(n1):
                for b in range(n2):
                    if idx % step == 0 and count < k:
                        r1 = layer_rows[i][min(a, len(layer_rows[i])-1)]
                        r2 = layer_rows[i+1][min(b, len(layer_rows[i+1])-1)]
                        edges_to_draw.append((r1 + pad_top, c1, r2 + pad_top, c2))
                        count += 1
                    idx += 1

    # Simple line drawer between (r1,c1) and (r2,c2)
    def draw_line(r1, c1, r2, c2, ch=edge_char):
        steps = max(abs(c2 - c1), abs(r2 - r1))
        if steps == 0:
            return
        for s in range(steps + 1):
            t = s / steps
            r = round(r1 + (r2 - r1) * t)
            c = round(c1 + (c2 - c1) * t)
            if canvas[r][c] == " ":
                canvas[r][c] = ch

    for (r1, c1, r2, c2) in edges_to_draw:
        draw_line(r1, c1, r2, c2)

    # Convert canvas to text
    lines = ["".join(row).rstrip() for row in canvas]

    # Legend
    if show_legend:
        sizes_str = " × ".join(str(s) for s in sizes)
        meta = []
        if original_max > rows:
            meta.append(f"scaled to height {rows} from max layer {original_max}")
        if sampled:
            meta.append(f"edges sampled ({len(edges_to_draw)}/{total_edges})")
        legend = f"[layers: {sizes_str}" + (f" | {'; '.join(meta)}" if meta else "") + "]"
        lines.append(legend)

    return "\n".join(lines)

# Demo
print(ascii_nn({'input':2,'hidden_1':4,'output':1}, max_height=12, edge_cap=500))

