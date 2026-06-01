"""
Launch all four services in parallel using the current Python interpreter.
Press Ctrl+C to stop everything.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_SERVICES = {
    "ingestion": [sys.executable, "-m", "src.ingestion.scheduler"],
    "engine":    [sys.executable, "-m", "src.engine.runner"],
    "api":       [sys.executable, "-m", "src.api.server"],
    "dashboard": [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
                  "--server.port", "8501", "--server.headless", "true"],
}


def _free_port(port: int) -> None:
    """Kill any process listening on the given port."""
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    for pid in result.stdout.strip().splitlines():
        subprocess.run(["kill", "-9", pid], capture_output=True)


def _start(name: str) -> subprocess.Popen:
    p = subprocess.Popen(_SERVICES[name], cwd=str(PROJECT_ROOT))
    print(f"  [{p.pid}] {name} started")
    return p


def main() -> None:
    print("Freeing ports 8000 and 8501...")
    _free_port(8000)
    _free_port(8501)
    time.sleep(1)

    print("Starting Signal Engine — press Ctrl+C to stop all services\n")
    procs: dict[str, subprocess.Popen] = {name: _start(name) for name in _SERVICES}

    print()
    print("  Dashboard  →  http://localhost:8501")
    print("  API docs   →  http://localhost:8000/docs")
    print()

    try:
        while True:
            for name, p in list(procs.items()):
                if p.poll() is not None:
                    print(f"  [!] {name} exited (code {p.returncode}) — restarting")
                    procs[name] = _start(name)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("All services stopped.")


if __name__ == "__main__":
    main()
