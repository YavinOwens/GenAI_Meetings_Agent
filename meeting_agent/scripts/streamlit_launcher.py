"""
Streamlit Launcher for Meeting Agent GUI

Launches the Streamlit web application interface for Meeting Agent.
Requires: pip install meeting-agent[StreamlitUI]
"""

import sys
from pathlib import Path


def main():
    """Launch the Streamlit Meeting Agent application."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("ERROR: Streamlit is not installed.")
        print("Please install it with: pip install meeting-agent[StreamlitUI]")
        print("Or: pip install streamlit>=1.28.0")
        sys.exit(1)

    # Get the path to streamlit_app.py
    app_path = Path(__file__).parent.parent / "streamlit_app.py"

    if not app_path.exists():
        print(f"ERROR: Streamlit app not found at {app_path}")
        sys.exit(1)

    # Launch Streamlit
    sys.argv = ["streamlit", "run", str(app_path)]
    stcli.main()


if __name__ == "__main__":
    main()
