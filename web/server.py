import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SCRIPT = BASE_DIR / "run_cuda_experiments.py"
WEB_DIR = BASE_DIR / "web"

tasks: dict[str, dict] = {}


def _kill_proc_tree(pid):
    import ctypes
    if os.name == "nt":
        os.system(f'taskkill /F /T /PID {pid} 2>nul')
    else:
        os.kill(pid, 15)


class Handler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            task_id = uuid.uuid4().hex[:12]
            tasks[task_id] = {
                "status": "running", "output": [], "result": None,
                "progress": None, "proc": None,
            }
            t = threading.Thread(target=self._run_task, args=(task_id, body), daemon=True)
            t.start()
            self._json({"task_id": task_id})
        elif parsed.path.startswith("/pause/"):
            task_id = parsed.path.split("/")[-1]
            info = tasks.get(task_id)
            if not info or info["status"] != "running":
                self._json({"error": "cannot pause"}, 400)
            else:
                info["status"] = "paused"
                info["output"].append("实验已暂停")
                self._json({"status": "paused"})
        elif parsed.path.startswith("/resume/"):
            task_id = parsed.path.split("/")[-1]
            info = tasks.get(task_id)
            if not info or info["status"] != "paused":
                self._json({"error": "cannot resume"}, 400)
            else:
                info["status"] = "running"
                info["output"].append("实验已恢复")
                self._json({"status": "running"})
        elif parsed.path.startswith("/stop/"):
            task_id = parsed.path.split("/")[-1]
            info = tasks.get(task_id)
            if not info or not info.get("proc") or info["proc"].poll() is not None:
                if info:
                    info["status"] = "stopped"
                self._json({"status": "stopped"})
            else:
                info["status"] = "stopping"
                _kill_proc_tree(info["proc"].pid)
                info["proc"].kill()
                info["output"].append("实验已停止")
                self._json({"status": "stopped"})
        elif parsed.path == "/resume":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            csv_path = body.get("csv_path")
            if not csv_path or not os.path.exists(csv_path):
                self._json({"error": "csv file not found"}, 400)
            else:
                task_id = uuid.uuid4().hex[:12]
                tasks[task_id] = {
                    "status": "running", "output": [], "result": None,
                    "progress": None, "proc": None,
                }
                body["resume_csv"] = csv_path
                t = threading.Thread(target=self._run_task, args=(task_id, body), daemon=True)
                t.start()
                self._json({"task_id": task_id})
        else:
            self._json({"error": "not found"}, 404)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/list-results":
            results = []
            if RESULTS_DIR.is_dir():
                for d in sorted(RESULTS_DIR.iterdir()):
                    if d.is_dir():
                        csv_file = d / "experiment_data.csv"
                        json_file = d / "experiment_params.json"
                        results.append({
                            "dir": str(d),
                            "name": d.name,
                            "csv": str(csv_file) if csv_file.exists() else None,
                            "params": str(json_file) if json_file.exists() else None,
                        })
            self._json({"results": results})
            return

        if parsed.path.startswith("/status/"):
            task_id = parsed.path.split("/")[-1]
            info = tasks.get(task_id)
            if not info:
                self._json({"error": "task not found"}, 404)
                return
            self._json({
                "status": info["status"],
                "output": info["output"][-100:],
                "progress": info.get("progress"),
                "result": info["result"],
            })
            return

        if parsed.path == "/":
            self.path = "/index.html"

        if parsed.path.startswith("/results/"):
            self.directory = str(RESULTS_DIR.parent)
            return super().do_GET()

        self.directory = str(WEB_DIR)
        return super().do_GET()

    def _run_task(self, task_id: str, params: dict):
        info = tasks[task_id]
        try:
            timestamp = "--timestamp" if params.get("timestamp", True) else ""
            args = [
                sys.executable, "-u", str(SCRIPT),
                "--aspect-ratios", params.get("aspect_ratios", "1.0"),
                "--min-cells", str(params.get("min_cells", 16)),
                "--max-cells", str(params.get("max_cells", 256)),
                "--cell-step", str(params.get("cell_step", 24)),
                "--min-density", str(params.get("min_density", 0.10)),
                "--max-density", str(params.get("max_density", 0.35)),
                "--density-step", str(params.get("density_step", 0.05)),
                "--trials", str(params.get("trials", 100)),
                "--seed", str(params.get("seed", 2026)),
                "--output-dir", str(RESULTS_DIR),
            ]
            if not params.get("interactive", True):
                args.append("--no-interactive")
            if timestamp:
                args.append("--timestamp")
            if params.get("vram_monitor"):
                args.append("--vram-monitor")
            resume_csv = params.get("csv_path") or params.get("resume_csv")
            if resume_csv and os.path.exists(resume_csv):
                args.extend(["--resume", resume_csv])

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            info["progress"] = {
                "pct": 0, "done_games": 0, "total_games": 0,
                "done_cfgs": 0, "total_cfgs": 0,
                "current_ar": "", "current_cells": "", "current_density": "",
            }
            create_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, bufsize=1, creationflags=create_flags,
            )
            info["proc"] = proc
            for line in proc.stdout:
                line = line.rstrip()
                if info["status"] == "paused":
                    while info["status"] == "paused":
                        time.sleep(0.5)
                if line.startswith("PROGRESS|"):
                    try:
                        parts = line.split("|")
                        for part in parts[1:]:
                            if "=" in part:
                                k, v = part.split("=", 1)
                                info["progress"][k] = v
                    except Exception:
                        pass
                else:
                    info["output"].append(line)

            proc.wait()
            if info["status"] in ("stopped", "stopping"):
                info["status"] = "stopped"
                return
            info["status"] = "done" if proc.returncode == 0 else "error"

            if proc.returncode == 0:
                results = self._find_results(RESULTS_DIR)
                info["result"] = results
            else:
                info["output"].append(f"进程退出码: {proc.returncode}")

        except Exception as e:
            if info["status"] != "stopping":
                info["status"] = "error"
                info["output"].append(f"错误: {e}")

    def _find_results(self, base: Path) -> dict | None:
        if not base.is_dir():
            return None
        dirs = sorted([p for p in base.iterdir() if p.is_dir()])
        for d in reversed(dirs):
            pngs = list(d.glob("*.png")) + list(d.glob("*.html"))
            if pngs:
                csv_file = d / "experiment_data.csv"
                return {
                    "dir": str(d),
                    "csv": str(csv_file) if csv_file.exists() else None,
                    "charts": sorted(str(f) for f in pngs),
                }
        pngs = list(base.glob("*.png")) + list(base.glob("*.html"))
        csv_file = base / "experiment_data.csv"
        if not pngs and not csv_file.exists():
            return None
        return {
            "dir": str(base),
            "csv": str(csv_file) if csv_file.exists() else None,
            "charts": sorted(str(f) for f in pngs),
        }

    def _json(self, data: dict, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        if args and isinstance(args[0], str) and "/status/" in args[0]:
            return
        super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"启动 CUDA 实验服务: http://localhost:{port}")
    print(f"结果目录: {RESULTS_DIR}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
