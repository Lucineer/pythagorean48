#!/usr/bin/env python3
"""
p48-weaver.py — Fleet Intelligence Weaver

Bridges three systems into one coordinated intelligence loop:

  warp-room (C P48 classifier) → p48-index-server (SHM vector store)
  → plato-server (tile DB) → plato-mythos (MUD embedding) 
  → edge-gateway (native inference) → fleet bottles

The weaving process:
  1. Monitor: Check all 4 services are healthy
  2. Sync: Rebuild P48 index from plato-server tiles
  3. Classify: Route each new tile through warp-room P48 for room assignment
  4. Embed: Run tile text through native inference for mythos-style latent
  5. Report: Write findings as fleet bulletins

This is the "fleet brain" — the coordination layer between all edge services.
"""

import json, os, sys, time, math, re, subprocess, urllib.request, urllib.parse, socket

# Service addresses
SERVICES = {
    "edge-gateway": "http://localhost:11435",
    "p48-index": "http://localhost:8846",
    "plato-server": "http://localhost:8847",
    "evennia-mud": ("localhost", 4000),
    "ollama": "http://localhost:11434",
    "warp-room": "/tmp/warp-room/warp-room",
}

def check_health():
    """Probe all services, return status dict."""
    status = {}
    
    # edge-gateway
    try:
        resp = urllib.request.urlopen(f"{SERVICES['edge-gateway']}/v1/native", timeout=2)
        data = json.loads(resp.read())
        status["edge-gateway"] = {
            "ok": data.get("available", False),
            "tps": data.get("tps", 0),
            "loaded": data.get("loaded", False),
            "backend": data.get("backend", "unknown"),
        }
    except Exception as e:
        status["edge-gateway"] = {"ok": False, "error": str(e)}
    
    # p48-index
    try:
        resp = urllib.request.urlopen(f"{SERVICES['p48-index']}/status", timeout=2)
        data = json.loads(resp.read())
        status["p48-index"] = {
            "ok": True,
            "vectors": data.get("n_vectors", 0),
        }
    except Exception as e:
        status["p48-index"] = {"ok": False, "error": str(e)}
    
    # plato-server
    try:
        resp = urllib.request.urlopen(f"{SERVICES['plato-server']}/", timeout=2)
        data = json.loads(resp.read())
        status["plato-server"] = {
            "ok": True,
            "tiles": data.get("tiles", 0),
            "rooms": data.get("rooms", 0),
        }
    except Exception as e:
        status["plato-server"] = {"ok": False, "error": str(e)}
    
    # Ollama
    try:
        resp = urllib.request.urlopen(f"{SERVICES['ollama']}/api/tags", timeout=2)
        data = json.loads(resp.read())
        models = [m.get("name", "?") for m in data.get("models", [])]
        status["ollama"] = {
            "ok": True,
            "models": models,
            "n_models": len(models),
        }
    except Exception as e:
        status["ollama"] = {"ok": False, "error": str(e)}
    
    # warp-room binary
    warp_ok = os.path.exists(SERVICES["warp-room"])
    status["warp-room"] = {
        "ok": warp_ok,
        "path": SERVICES["warp-room"],
    }
    
    return status


def warp_room_classify(text):
    """Route text through warp-room's P48 classifier."""
    if not os.path.exists(SERVICES["warp-room"]):
        return None
    try:
        result = subprocess.run(
            [SERVICES["warp-room"], "--infer", text],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except:
        pass
    return None


def warp_room_classify_neon(text):
    """Route text through warp-room's NEON P48 classifier."""
    if not os.path.exists(SERVICES["warp-room"]):
        return None
    try:
        result = subprocess.run(
            [SERVICES["warp-room"], "--infer-neon", text],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except:
        pass
    return None


def native_infer(prompt, max_tokens=50):
    """Infer via native socket (fastest path)."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(15)
        s.connect("/tmp/edge-native.sock")
        req = json.dumps({"prompt": prompt, "max_tokens": max_tokens}) + "\n"
        s.send(req.encode())
        resp = b""
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            resp += chunk
            if b"\n" in resp: break
        s.close()
        return json.loads(resp.decode().strip())
    except Exception as e:
        return {"text": "", "tokens": 0, "tps": 0, "error": str(e)}


def native_infer_http(messages, max_tokens=100):
    """Infer via HTTP gateway (chat template applied)."""
    try:
        data = json.dumps({
            "model": "deepseek-r1:1.5b",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{SERVICES['edge-gateway']}/v1/chat/completions?native=true",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=20)
        result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["completion_tokens"]
        tps = result.get("tps", 0)
        return {"text": text, "tokens": tokens, "tps": tps, "backend": "native"}
    except Exception as e:
        return {"text": "", "tokens": 0, "tps": 0, "error": str(e)}


def weave_loop(interval=300):
    """Run the weaving loop."""
    print(f"Fleet Intelligence Weaver starting (interval={interval}s)")
    print(f"  warp-room:     {SERVICES['warp-room']}")
    print(f"  edge-gateway:  {SERVICES['edge-gateway']}")
    print(f"  p48-index:     {SERVICES['p48-index']}")
    print(f"  plato-server:  {SERVICES['plato-server']}")
    print()
    
    while True:
        tick_start = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Weave tick...")
        
        # 1. Health check
        health = check_health()
        services_up = sum(1 for h in health.values() if h.get("ok", False))
        services_total = len(health)
        print(f"  Health: {services_up}/{services_total} services up")
        
        for name, h in health.items():
            if h.get("ok"):
                extra = ""
                if "tps" in h: extra = f", {h['tps']} t/s"
                if "vectors" in h: extra = f", {h['vectors']} vectors"
                if "tiles" in h: extra = f", {h['tiles']} tiles"
                if "models" in h: extra = f", {h['n_models']} models"
                print(f"    |g{name}|n: up{extra}")
            else:
                print(f"    |r{name}|n: {h.get('error', 'unknown')}")
        
        # 2. Test native inference
        infer = native_infer("hello from weave", max_tokens=10)
        if infer.get("tokens", 0) > 0:
            print(f"  Native infer: {infer['tokens']} tokens @ {infer['tps']} t/s")
        
        # 3. Test warp-room P48 classification
        for test_text in ["gpu nvidia cuda kernel", "fleet deadman agent"]:
            result = warp_room_classify_neon(test_text)
            if result:
                room = result.get("room", "?")
                print(f"  P48 NEON: \"{test_text}\" → {room}")
        
        # 4. Test P48 index search
        try:
            resp = urllib.request.urlopen(
                f"{SERVICES['p48-index']}/search?q=fleet+deadman&top_k=1",
                timeout=3
            )
            data = json.loads(resp.read())
            results = data.get("results", [])
            if results:
                r = results[0]
                q = r.get("tile", {}).get("question", "?")[:40]
                print(f"  P48 search 'fleet deadman': {r['distance']} → {q}")
        except Exception:
            pass
        
        elapsed = time.time() - tick_start
        wait = max(1, interval - int(elapsed))
        print(f"  Tick done in {elapsed:.1f}s, next in {wait}s")
        
        print(".")  # Marker
        time.sleep(wait)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        # One-shot health check
        health = check_health()
        print(json.dumps(health, indent=2))
        
        infer = native_infer_http([{"role": "user", "content": "What's your name?"}], max_tokens=20)
        if infer.get("text"):
            print(f"\nNative: {infer['text'][:80]}")
        
        for test in ["gpu cuda nvidia", "fleet deadman bottle"]:
            r = warp_room_classify_neon(test)
            if r:
                print(f"Warp-room NEON: '{test}' → {r.get('room', '?')}")
    else:
        interval = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
        weave_loop(interval)
