import subprocess
import sys
import time


def main():
    print("Starting FastAPI backend (http://127.0.0.1:8000)...")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000"]
    )

    time.sleep(2)  # Wait for API to initialize

    print("Starting Streamlit dashboard...")
    dashboard = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    )

    try:
        dashboard.wait()
        api.wait()
    except KeyboardInterrupt:
        print("\nShutting down both services...")
        dashboard.terminate()
        api.terminate()


if __name__ == "__main__":
    main()
