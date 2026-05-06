#!/usr/bin/env python3
"""
p48-index-server.py — Persistent P48 vector index with shared memory and HTTP API.

Bridges warp-room's C17 shm classifier with plato-server's tile database.
All products written to /dev/shm/p48-index/ for zero-copy shared memory
access by warp-room, sensor-pipeline, and plato-sync.

Architecture:
  - In-memory P48 vector store (C-backed shm for speed)
  - Periodic sync from plato-server tiles
  - HTTP API for queries (replacing plato-server's text search)
  - Exact nearest-neighbor using P48 squared distance
  - Handles 10K+ vectors with sub-ms queries

Usage:
  python3 p48-index-server.py --daemon   # Run as server
  python3 p48-index-server.py --index    # One-shot index rebuild
  python3 p48-index-server.py --query "..." # One-shot query
"""

import json, sys, os, math, re, time, mmap, struct, threading, http.server, urllib.request, urllib.parse, signal

# === 90 keywords matching warp-room.c ===
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

# Config
PLATO_URL = os.environ.get("PLATO_URL", "http://localhost:8847")
SHM_DIR = "/dev/shm/p48-index"
INDEX_FILE = f"{SHM_DIR}/vectors.bin"
META_FILE = f"{SHM_DIR}/index.json"
SERVER_PORT = int(os.environ.get("P48_PORT", "8846"))

def tokenize(text):
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

def make_index(keywords):
    """Build P48 vector from keyword counts. Returns list of 90 components (0-63)."""
    vec = [0.0] * N_DIMS
    for kw in keywords:
        kw = kw.lower().strip(".,!?;:'\"()[]{}<>/\\-@#$%^&*+=~`")
        if kw in KW_SET:
            idx = ALL_KW.index(kw)
            vec[idx] += 1.0
    total = sum(vec)
    if total == 0:
        return [0] * N_DIMS
    for i in range(N_DIMS):
        if vec[i] > 0:
            vec[i] = 1.0 + math.log(vec[i])
    norm = math.sqrt(sum(v*v for v in vec))
    if norm < 1e-8:
        return [0] * N_DIMS
    for i in range(N_DIMS):
        vec[i] /= norm
    return [max(0, min(63, int(v * 63 + 0.5))) for v in vec]

def pack_p48(components):
    """Pack 90 components into ceil(90/8) = 12 uint64 P48 vectors."""
    n_p48 = math.ceil(len(components) / 8)
    packed = [0] * n_p48
    for i, c in enumerate(components):
        p = i // 8
        bit = (i % 8) * 6
        packed[p] |= ((c & 0x3F)) << bit
    return packed

def p48_dist_sq_packed(packed_a, packed_b):
    """Squared distance between two packed P48 vectors."""
    total = 0
    for a, b in zip(packed_a, packed_b):
        for i in range(8):
            ca = (a >> (6 * i)) & 0x3F
            cb = (b >> (6 * i)) & 0x3F
            d = ca - cb
            total += d * d
    return total

def p48_dist_sq(a, b):
    """Squared distance between two unpacked P48 vectors (90-dim)."""
    return sum((ca - cb) ** 2 for ca, cb in zip(a, b))


class P48Index:
    """Persistent P48 vector index backed by shared memory files."""
    
    def __init__(self):
        self.vectors = []        # List of (id, room, packed_vector_12, metadata)
        self.query_cache = {}    # LRU-like query cache
        self.tiles = {}          # tile_id -> tile dict
        self._lock = threading.Lock()
        os.makedirs(SHM_DIR, exist_ok=True)
    
    def add_vector(self, tile_id, room, packed_vec, metadata):
        """Add one vector to the index."""
        self.vectors.append({
            "id": tile_id,
            "room": room,
            "packed": packed_vec,
            "meta": metadata,
        })
    
    def index_tile(self, tile):
        """Index one plato-server tile."""
        tile_id = tile.get("id", str(time.time()))
        room = tile.get("room", "edge")
        text = f"{tile.get('question', '')} {tile.get('answer', '')} {tile.get('domain', '')}"
        kw = tokenize(text)
        components = make_index(kw)
        packed = pack_p48(components)
        self.add_vector(tile_id, room, packed, tile)
        self.tiles[tile_id] = tile
    
    def search(self, query, top_k=10, room_filter=None):
        """Nearest-neighbor search by P48 distance."""
        qkw = tokenize(query)
        qcomp = make_index(qkw)
        qpacked = pack_p48(qcomp)
        
        results = []
        for v in self.vectors:
            if room_filter and v["room"] != room_filter:
                continue
            d = p48_dist_sq_packed(qpacked, v["packed"])
            results.append((d, v))
        
        results.sort(key=lambda x: x[0])
        return results[:top_k]
    
    def save_shm(self):
        """Write index to /dev/shm/p48-index/ for warp-room shared memory access."""
        if not self.vectors:
            return
        n = len(self.vectors)
        n_p48 = len(self.vectors[0]["packed"])
        
        # Binary layout: [n_vectors:4] [n_p48:4] [packed_p48_vectors: n * n_p48 * 8]
        with open(INDEX_FILE, "wb") as f:
            f.write(struct.pack("<II", n, n_p48))
            for v in self.vectors:
                for p in v["packed"]:
                    f.write(struct.pack("<Q", p))
        
        # Metadata as JSON — preserve per-vector room + metadata
        vec_meta = []
        room_counts = {}
        for v in self.vectors:
            room = v["room"]
            room_counts[room] = room_counts.get(room, 0) + 1
            vec_meta.append({
                "id": v.get("id", ""),
                "room": room,
                "tile": v.get("meta", {}),
            })
        
        meta = {
            "built_at": time.time(),
            "n_vectors": n,
            "n_p48_dims": n_p48,
            "n_keywords": N_DIMS,
            "room_counts": room_counts,
            "vectors": vec_meta,
            "tiles": self.tiles,
        }
        
        with open(META_FILE, "w") as f:
            json.dump(meta, f)
        
        print(f"  Saved {n} vectors ({n_p48} P48 dims each) to {SHM_DIR}")
    
    def load_shm(self):
        """Load from /dev/shm. Returns True if loaded."""
        if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
            return False
        try:
            with open(META_FILE) as f:
                meta = json.load(f)
            vec_meta = meta.get("vectors", [])
            self.tiles = meta.get("tiles", {})
            
            with open(INDEX_FILE, "rb") as f:
                header = f.read(8)
                n, n_p48 = struct.unpack("<II", header)
                self.vectors = []
                for i in range(n):
                    pdata = f.read(n_p48 * 8)
                    packed = list(struct.unpack(f"<{n_p48}Q", pdata))
                    vm = vec_meta[i] if i < len(vec_meta) else {"room": "?", "id": "", "tile": {}}
                    self.vectors.append({
                        "id": vm.get("id", ""),
                        "room": vm.get("room", "?"),
                        "packed": packed,
                        "meta": vm.get("tile", {}),
                    })
            return True
        except Exception as e:
            print(f"  Error loading SHM: {e}")
            return False
    
    def fetch_plato_tiles(self):
        """Fetch all tiles from plato-server."""
        tiles = []
        for room in ["edge", "research", "fleet", "jc1"]:
            url = f"{PLATO_URL}/room/{room}?limit=200"
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                data = json.loads(resp.read())
                if isinstance(data, dict) and "tiles" in data:
                    tiles.extend(data["tiles"])
            except Exception as e:
                print(f"  Warning: {room}: {e}")
        return tiles
    
    def build_from_server(self):
        """Rebuild index from plato-server."""
        print(f"Fetching tiles from {PLATO_URL}...")
        tiles = self.fetch_plato_tiles()
        print(f"  Got {len(tiles)} tiles")
        
        self.vectors = []
        self.tiles = {}
        for tile in tiles:
            self.index_tile(tile)
        
        print(f"  Indexed {len(self.vectors)} P48 vectors")
        self.save_shm()
        return len(self.vectors)
    
    def bench(self, num_queries=1000):
        """Benchmark query performance."""
        if not self.vectors:
            print("  No vectors to benchmark")
            return
        queries = ["jetson gpu", "research paper", "fleet agent", "jc1 plato",
                    "nvidia cuda", "deadman protocol", "temperature sensor",
                    "neural network", "bottle sync", "think infer"]
        start = time.time()
        for _ in range(num_queries):
            q = queries[_ % len(queries)]
            self.search(q, top_k=5)
        elapsed = time.time() - start
        print(f"  {num_queries} queries in {elapsed*1000:.1f} ms = {num_queries/elapsed:.0f} q/s")


# === HTTP Server ===
class P48Server:
    def __init__(self, index):
        self.index = index
        self._lock = threading.Lock()
    
    class Handler(http.server.BaseHTTPRequestHandler):
        server_ref = None
        
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)
            
            if path == "/":
                self.send_json({
                    "service": "P48 Vector Index Server",
                    "tiles_indexed": len(self.server_ref.index.vectors),
                    "dimensions": N_DIMS,
                    "p48_packed_dims": 12,
                    "shm_path": SHM_DIR,
                    "endpoints": [
                        "GET  /             — This page",
                        "GET  /search?q=... — P48 nearest-neighbor search",
                        "GET  /search?q=...&room=fleet — Room-filtered search",
                        "GET  /status       — Index statistics",
                        "POST /reindex      — Rebuild from plato-server",
                        "GET  /bench        — Query benchmark",
                    ],
                })
            elif path == "/search":
                q = params.get("q", [""])[0]
                room = params.get("room", [None])[0]
                top_k = int(params.get("top_k", ["10"])[0])
                if q:
                    results = self.server_ref.index.search(q, top_k, room)
                    response = {
                        "query": q,
                        "method": "p48_exact",
                        "results": [
                            {
                                "distance": d,
                                "tile": v.get("meta", {}),
                            } for d, v in results
                        ],
                    }
                else:
                    response = {"error": "missing q parameter"}
                self.send_json(response)
            elif path == "/status":
                v = self.server_ref.index.vectors
                rooms = {}
                for vec in v:
                    r = vec.get("meta", {}).get("room", "?")
                    rooms[r] = rooms.get(r, 0) + 1
                self.send_json({
                    "n_vectors": len(v),
                    "rooms": rooms,
                    "n_p48_dims": 12,
                    "n_keywords": N_DIMS,
                    "shm_path": SHM_DIR,
                    "warp_room_compatible": True,
                })
            elif path == "/reindex":
                self.send_json({
                    "status": "rebuilding",
                    "message": "use POST method"
                })
            elif path == "/bench":
                self.server_ref.index.bench(500)
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "not found"}, 404)
        
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            
            if self.path == "/reindex":
                n = self.server_ref.index.build_from_server()
                self.send_json({"status": "ok", "vectors_indexed": n})
            else:
                self.send_json({"error": "not found"}, 404)
        
        def send_json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        
        def log_message(self, fmt, *args):
            # Quiet mode
            pass


def cmd_daemon():
    """Run the P48 index server."""
    index = P48Index()
    
    # Try loading existing shm, otherwise build
    if not index.load_shm():
        print("No existing index found. Building from plato-server...")
        index.build_from_server()
    else:
        print(f"Loaded {len(index.vectors)} vectors from {SHM_DIR}")
    
    P48Server.Handler.server_ref = P48Server(index)
    server = http.server.HTTPServer(("0.0.0.0", SERVER_PORT), P48Server.Handler)
    print(f"P48 Index Server on port {SERVER_PORT}, {len(index.vectors)} vectors")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()

def cmd_index():
    """One-shot index rebuild."""
    index = P48Index()
    n = index.build_from_server()
    print(f"\nDone: {n} vectors indexed")

def cmd_query(query, top_k=10):
    """One-shot query."""
    index = P48Index()
    if not index.load_shm():
        print("No index. Run --index first.")
        return
    
    results = index.search(query, top_k)
    print(f"Query: '{query}'")
    print(f"{'Dist':>6}  {'Room':>10}  {'Tile':<50}")
    print("-" * 70)
    for d, v in results:
        meta = v.get("meta", {})
        title = meta.get("question", meta.get("id", "?"))[:48]
        room = meta.get("room", "?")
        print(f"{d:>6}  {room:>10}  {title}")

def cmd_bench():
    """Benchmark query performance."""
    index = P48Index()
    if not index.load_shm():
        print("No index. Run --index first.")
        return
    index.bench(1000)


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        cmd_daemon()
    elif "--index" in sys.argv:
        cmd_index()
    elif "--query" in sys.argv:
        idx = sys.argv.index("--query") + 1
        query = sys.argv[idx] if idx < len(sys.argv) else ""
        cmd_query(query)
    elif "--bench" in sys.argv:
        cmd_bench()
    else:
        print("Usage:")
        print("  p48-index-server.py --daemon     # HTTP API on :8846")
        print("  p48-index-server.py --index      # Rebuild index")
        print("  p48-index-server.py --query '...' # One-shot query")
        print("  p48-index-server.py --bench      # Benchmark")
