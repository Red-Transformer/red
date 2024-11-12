import subprocess

subprocess.run(
    ["python -m http.server & disown ; exit"],
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

