#!/usr/bin/env python3
"""
p48-tile-search.py — P48 exact nearest-neighbor tile search for plato-server.

Replaces plato-server's simple text search with exact integer distance
using Pythagorean48 6-bit vector encoding.

Architecture:
  1. Index phase: extract keywords from each tile, quantize to P48 vector
  2. Query phase: extract keywords from query, quantize, find nearest P48 vector
  3. Returns tiles ranked by exact integer distance (lower = better match)

Integration with warp-room:
  - Same 90-keyword vocabulary across 4 rooms (edge/research/fleet/jc1)
  - Same L2-normalized P48 encoding
  - Tiles are indexed at startup and updated when new tiles arrive

Usage:
  python3 p48-tile-search.py --index     # Build P48 index from all tiles
  python3 p48-tile-search.py --query "..." # Search tiles by exact distance
  python3 p48-tile-search.py --serve     # Serve as local API on port 8848
"""

import json, sys, os, math, re, time, http.server

PLATO_URL = os.environ.get("PLATO_URL", "http://localhost:8847")

# ---- 90 keywords matching warp-room.c ---- #
EDGE_KW = ["jetson","cpu","gpu","memory","temperature","load","uptime",
           "disk","thermal","fan","power","nvidia","cuda","nvcc",
           "arm64","aarch64","swap","network","interface","sensor",
           "telemetry","hardware","clock","throttle","edge","device"]
RESEARCH_KW = ["research","paper","study","findings","analysis","experiment",
               "benchmark","performance","test","comparison","evaluation",
               "learn","training","dataset","model","inference","llm",
               "neural","embedding","vector","similarity","tile",
               "investigation","methodology","result","conclusion","algorithm"]
FLEET_KW = ["fleet","agent","oracle","forge","vessel","bottle","matrix",
            "heartbeat","sync","mesh","iron","coordination","bridge",
            "pki","cert","trust","deadman","migration","protocol",
            "lighthouse","beacon","dm","conduit","message"]
JC1_KW = ["jc1","jetsonclaw","plato","evennia","flato","mythos",
          "cocapn","libllama","gguf","sovereign","infer","think","vessel"]

ALL_KW = EDGE_KW + RESEARCH_KW + FLEET_KW + JC1_KW
KW_SET = set(ALL_KW)
N_DIMS = len(ALL_KW)  # 90

def make_index(keywords, n_dims=N_DIMS):
    """Build P48 vector from keyword counts. Returns list of P48 components (0-63)."""
    vec = [0.0] * n_dims
    for kw in keywords:
        kw = kw.lower().strip(".,!?;:'\"()[]{}<>/\\-@#$%^&*+=~`")
        if kw in KW_SET:
            idx = ALL_KW.index(kw)
            vec[idx] += 1.0
    
    # TF weighting
    total = sum(vec)
    if total == 0:
        return [0] * n_dims
    
    for i in range(n_dims):
        if vec[i] > 0:
            vec[i] = 1.0 + math.log(vec[i])
    
    # L2 normalize
    norm = math.sqrt(sum(v*v for v in vec))
    if norm < 1e-8:
        return [0] * n_dims
    for i in range(n_dims):
        vec[i] /= norm
    
    # Quantize to 6 bits (0-63) — matching warp-room float_to_p48_component
    return [max(0, min(63, int(v * 63 + 0.5))) for v in vec]

def p48_dist_sq(a, b):
    """Exact integer squared distance between two P48 vectors."""
    return sum((ca - cb) ** 2 for ca, cb in zip(a, b))

def tokenize(text):
    """Split text into lowercase words, removing punctuation."""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

class P48TileSearch:
    """P48-powered tile search engine."""
    
    def __init__(self):
        self.tiles = []      # list of tile dicts
        self.vectors = []     # list of P48 vectors (90-dim each)
        self.n_dims = N_DIMS
    
    def index_tile(self, tile):
        """Add a single tile to the index."""
        # Extract keywords from question + answer + room
        text = f"{tile.get('question', '')} {tile.get('answer', '')} {tile.get('room', '')}"
        kw = tokenize(text)
        vec = make_index(kw)
        self.tiles.append(tile)
        self.vectors.append(vec)
    
    def index_all(self, tiles_list):
        """Index all tiles."""
        for tile in tiles_list:
            self.index_tile(tile)
    
    def search(self, query, top_k=10):
        """Search tiles by exact P48 distance. Returns list of (distance, tile) tuples."""
        qkw = tokenize(query)
        qvec = make_index(qkw)
        
        results = []
        for i, tile in enumerate(self.tiles):
            d = p48_dist_sq(qvec, self.vectors[i])
            results.append((d, tile))
        
        results.sort(key=lambda x: x[0])
        return results[:top_k]
    
    def fetch_all_tiles(self):
        """Fetch all tiles from plato-server using pagination."""
        import urllib.request
        tiles = []
        for room in ["edge", "research", "fleet", "jc1"]:
            offset = 0
            limit = 50
            while True:
                url = f"{PLATO_URL}/room/{room}?offset={offset}&limit={limit}"
                try:
                    resp = urllib.request.urlopen(url, timeout=5)
                    data = json.loads(resp.read())
                    if isinstance(data, list):
                        batch = data
                    elif isinstance(data, dict) and "tiles" in data:
                        batch = data["tiles"]
                    else:
                        break
                    if not batch:
                        break
                    tiles.extend(batch)
                    if len(batch) < limit:
                        break
                    offset += limit
                except Exception as e:
                    break
            # Fallback: get all via /tiles/recent
            try:
                resp = urllib.request.urlopen(f"{PLATO_URL}/tiles/recent?room={room}&limit=500", timeout=5)
                data = json.loads(resp.read())
                if isinstance(data, list):
                    tiles.extend(data)
            except:
                pass
        return tiles

def cmd_index():
    """Build P48 index from all tiles."""
    engine = P48TileSearch()
    tiles = engine.fetch_all_tiles()
    engine.index_all(tiles)
    
    # Save index
    index_data = {
        "n_dims": engine.n_dims,
        "tiles": engine.tiles,
        "vectors": engine.vectors,
        "built_at": time.time(),
    }
    with open("/tmp/p48-tile-index.json", "w") as f:
        json.dump(index_data, f)
    print(f"Indexed {len(tiles)} tiles, {engine.n_dims} dims each")
    print(f"Saved to /tmp/p48-tile-index.json")

def cmd_query(query, top_k=10):
    """Search tiles by P48 distance."""
    if not os.path.exists("/tmp/p48-tile-index.json"):
        print("No index found. Run --index first.")
        return
    
    with open("/tmp/p48-tile-index.json") as f:
        index = json.load(f)
    
    engine = P48TileSearch()
    engine.tiles = index["tiles"]
    engine.vectors = index["vectors"]
    engine.n_dims = index["n_dims"]
    
    results = engine.search(query, top_k)
    print(f"Query: '{query}'")
    print(f"{'Dist':>6}  {'Room':>10}  {'Tile':<50}")
    print("-" * 70)
    for d, tile in results:
        title = tile.get("question", "")[:48]
        room = tile.get("room", "?")
        print(f"{d:>6}  {room:>10}  {title}")

def cmd_serve():
    """Serve P48 search as HTTP API on port 8848."""
    # Load index
    if not os.path.exists("/tmp/p48-tile-index.json"):
        print("No index found. Run --index first.")
        return
    
    with open("/tmp/p48-tile-index.json") as f:
        index = json.load(f)
    
    engine = P48TileSearch()
    engine.tiles = index["tiles"]
    engine.vectors = index["vectors"]
    engine.n_dims = index["n_dims"]
    
    print(f"Serving P48 tile search on port 8848 ({len(engine.tiles)} tiles indexed)")
    
    class P48Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/search"):
                import urllib.parse
                parsed = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
                if q:
                    results = engine.search(q)
                    response = json.dumps({
                        "query": q,
                        "method": "p48_exact",
                        "results": [{"distance": d, "tile": t} for d, t in results]
                    })
                else:
                    response = json.dumps({"error": "missing q parameter"})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())
            else:
                response = json.dumps({
                    "service": "P48 Tile Search",
                    "tiles_indexed": len(engine.tiles),
                    "dimensions": engine.n_dims,
                    "endpoints": ["GET /search?q=..."],
                })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())
        
        def log_message(self, *a):
            pass
    
    server = http.server.HTTPServer(("0.0.0.0", 8848), P48Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    if "--index" in sys.argv:
        cmd_index()
    elif "--query" in sys.argv:
        idx = sys.argv.index("--query") + 1
        query = sys.argv[idx] if idx < len(sys.argv) else ""
        cmd_query(query)
    elif "--serve" in sys.argv:
        cmd_serve()
    else:
        print("Usage:")
        print("  p48-tile-search.py --index            # Index all tiles")
        print("  p48-tile-search.py --query \"text\"    # Search by P48 distance")
        print("  p48-tile-search.py --serve            # HTTP API on :8848")
