import subprocess
import sys

from fastapi import APIRouter

router = APIRouter()


@router.post("/train")
def trigger_training() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "src.train"],
        capture_output=True, text=True, timeout=300,
    )
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr,
    }
