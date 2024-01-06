import multiprocessing

from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

loglevel = "debug"

accesslog = LOG_DIR.joinpath("access.log").as_posix()
errorlog = LOG_DIR.joinpath("error.log").as_posix()
pidfile = LOG_DIR.joinpath("gunicorn.pid").as_posix()

bind = "127.0.0.1:8000"
proc_name = "ocr"

capture_output = True
debug = True
daemon = False

backlog = 2048
graceful_timeout = 300
timeout = 300
workers = multiprocessing.cpu_count()
worker_class = "uvicorn.workers.UvicornH11Worker"
worker_connections = 1000
x_forwarded_for_header = "X-FORWARDED-FOR"
