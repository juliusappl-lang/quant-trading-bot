import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])


if __name__ == "__main__":
    main()
