import os
import sys
import runpy
import subprocess
import threading
import time
import glob
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
# offline deps: container network blocks pypi; wheels are vendored in repo
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-index", "--no-deps", "--find-links",
                       os.path.join(ROOT, "vendor_wheels"),
                       "gymnasium", "farama-notifications", "cloudpickle",
                       "diffusers", "huggingface-hub", "filelock",
                       "importlib-metadata", "zipp", "packaging",
                       "typing-extensions", "regex", "tqdm", "safetensors",
                       "requests", "matplotlib", "contourpy", "cycler",
                       "fonttools", "kiwisolver", "pyparsing",
                       "python-dateutil", "six", "tensorboardX", "protobuf"])


def sh(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=e)


def start_checkpoint_exporter(exp_name, branch, keep_n=3):
    """Reliable checkpoint channel back to GitHub:
    1) copy new int_models files to SDK-blessed exported_data path
    2) git-plumbing force-push the newest `keep_n` checkpoints to `branch`
       (fresh root commit each time; never touches the working tree)"""
    run_dir = os.path.join(ROOT, "logs", exp_name, "exported_data",
                           time.strftime("%Y-%m-%d_%H-%M-%S"))
    staging = os.path.join(ROOT, "ckpt_git", exp_name)
    seen = set()
    dirty = {"v": False}
    tmp_index = os.path.join("/tmp", f"ckpt_index_{exp_name}")

    def export_once():
        nonlocal_local = None
        files = sorted(glob.glob(os.path.join(
            ROOT, "output", "int_models", "model_*.pt")),
            key=lambda f: int(os.path.basename(f)[6:-3]))
        changed = False
        for f in files:
            base = os.path.basename(f)
            if base in seen:
                continue
            seen.add(base)
            it = int(base[6:-3])
            os.makedirs(run_dir, exist_ok=True)
            shutil.copyfile(f, os.path.join(run_dir, f"model_{it}.pt"))
            changed = True
        if not changed:
            return
        os.makedirs(staging, exist_ok=True)
        for f in glob.glob(os.path.join(staging, "*.pt")):
            os.remove(f)
        for f in files[-keep_n:]:
            shutil.copyfile(f, os.path.join(staging, os.path.basename(f)))
        dirty["v"] = True

    def git_push():
        # build a tree from ONLY the staging dir via a temp index
        if os.path.exists(tmp_index):
            os.remove(tmp_index)
        env = {"GIT_INDEX_FILE": tmp_index}
        r = sh(["git", "read-tree", "--empty"], env=env)
        if r.returncode != 0:
            print(f"[exporter] read-tree rc={r.returncode}", flush=True)
            return False
        r = sh(["git", "add", "-f", "--all", staging], env=env)
        if r.returncode != 0:
            print(f"[exporter] add rc={r.returncode} {r.stderr[-200:]}",
                  flush=True)
            return False
        t = sh(["git", "write-tree"], env=env)
        if t.returncode != 0:
            print(f"[exporter] write-tree rc={t.returncode}", flush=True)
            return False
        tree = t.stdout.strip()
        ident = {"GIT_AUTHOR_NAME": "ckpt-exporter",
                 "GIT_AUTHOR_EMAIL": "trainer@gradmotion",
                 "GIT_COMMITTER_NAME": "ckpt-exporter",
                 "GIT_COMMITTER_EMAIL": "trainer@gradmotion"}
        ident.update(env)
        c = sh(["git", "commit-tree", tree, "-m",
                f"ckpts {time.strftime('%H:%M:%S')}"], env=ident)
        if c.returncode != 0:
            print(f"[exporter] commit-tree rc={c.returncode} "
                  f"{c.stderr[-200:]}", flush=True)
            return False
        commit = c.stdout.strip()
        sh(["git", "update-ref", f"refs/heads/{branch}", commit])
        p = sh(["git", "push", "--force", "origin", branch])
        ok = p.returncode == 0
        print(f"[exporter] git push {branch} tree={tree[:8]}: "
              f"rc={p.returncode} {'' if ok else p.stderr[-300:]}", flush=True)
        return ok

    def loop():
        # diagnostics: does origin carry credentials?
        r = sh(["git", "remote", "get-url", "origin"])
        url = r.stdout.strip()
        has_cred = ("@" in url.split("//")[-1]) if url else False
        print(f"[exporter] origin cred-embedded: {has_cred}", flush=True)
        while True:
            try:
                export_once()
                if dirty["v"]:
                    if git_push():
                        dirty["v"] = False
            except Exception as e:
                print(f"[exporter] error: {e}", flush=True)
            time.sleep(180)

    threading.Thread(target=loop, daemon=True).start()


start_checkpoint_exporter(os.environ.get("X1_EXP_NAME", "x1_smp"),
                          os.environ.get("X1_CKPT_BRANCH", "ckpt_x1_smp"))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/smp_x1_env.yaml",
            "--agent_config", "data/agents/smp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", "500000000"]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")

# natural end: push the final output/model.pt through the same channel
try:
    exp = os.environ.get("X1_EXP_NAME", "x1_smp")
    staging = os.path.join(ROOT, "ckpt_git", exp)
    os.makedirs(staging, exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, "output", "model.pt"),
                    os.path.join(staging, "model_final.pt"))
    print("[exporter] final model staged; pushing", flush=True)
    # reuse module-level funcs via closure is not possible here; do it inline
    tmp_index = os.path.join("/tmp", f"ckpt_index_{exp}_final")
    if os.path.exists(tmp_index):
        os.remove(tmp_index)
    env = {"GIT_INDEX_FILE": tmp_index}
    sh(["git", "read-tree", "--empty"], env=env)
    sh(["git", "add", "-f", "--all", staging], env=env)
    t = sh(["git", "write-tree"], env=env)
    if t.returncode == 0:
        ident = {"GIT_AUTHOR_NAME": "ckpt-exporter",
                 "GIT_AUTHOR_EMAIL": "trainer@gradmotion",
                 "GIT_COMMITTER_NAME": "ckpt-exporter",
                 "GIT_COMMITTER_EMAIL": "trainer@gradmotion"}
        ident["GIT_INDEX_FILE"] = tmp_index
        c = sh(["git", "commit-tree", t.stdout.strip(), "-m", "final model"],
               env=ident)
        if c.returncode == 0:
            branch = os.environ.get("X1_CKPT_BRANCH", "ckpt_x1_smp")
            sh(["git", "update-ref", f"refs/heads/{branch}", c.stdout.strip()])
            p = sh(["git", "push", "--force", "origin", branch])
            print(f"[exporter] final push rc={p.returncode} "
                  f"{p.stderr[-200:] if p.returncode else 'OK'}", flush=True)
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
