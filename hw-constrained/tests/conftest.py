import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.normpath(os.path.join(PKG, "..", "ppo-sb3-scripts")))

MOCK_RUNNER = os.path.join(PKG, "testing", "mock_ns3_runner.py")
