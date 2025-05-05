import subprocess
import os
import sys


def streamlit_app_cli():
    argv = sys.argv[1:]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(cur_dir, "streamlit_app.py")
    # 获取当前 Python 解释器的路径 (pipx 环境中的 python.exe)
    python_executable = sys.executable

    # 构建命令，使用 "python -m streamlit run ..."
    cmd = [
        python_executable,  # 使用当前环境的 Python
        "-m",               # 运行模块
        "streamlit",        # 要运行的模块名
        "run",              # streamlit 的子命令
        app_path,
        "--server.fileWatcherType", "none",
        "--server.headless", "true"
    ]
    if argv:
        # 将 marker_gui 的额外参数传递给 streamlit run
        cmd += ["--"] + argv

    # 执行命令, 添加 check=True 可以在失败时抛出更明确的错误
    try:
        subprocess.run(cmd, env={**os.environ, "IN_STREAMLIT": "true"}, check=True)
    except FileNotFoundError:
        print(f"Error: Could not find Python executable at {python_executable}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
