"""
ASD Early Screening Tool - STAT Digitization
Entry point for the application.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import ASDScreeningApp
import tkinter as tk


def main():
    root = tk.Tk()
    app = ASDScreeningApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
