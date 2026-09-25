"""CWD-safe AMP training launcher (platform executes startScript w/ python)."""
import os
import sys
import runpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/amp_x1_env.yaml",
            "--agent_config", "data/agents/amp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/"]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")
