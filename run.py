"""SafeDrive AI - development server entry point.

Usage:
    python run.py
Then open http://127.0.0.1:5000 in a browser.
"""

from src.api import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting SafeDrive AI at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
