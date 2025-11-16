import yagmail
import traceback
import subprocess
import sys
import datetime

'''This is how we run scripts on castor
    tmus new -s job
    Ctrl+B  then  D
    tmux attach -t job
    tmux kill-session -t job
'''

EMAIL = ""
APP_PASSWORD = ""   # your Gmail App Password

def notify(subject, body):
    try:
        yag = yagmail.SMTP(EMAIL, APP_PASSWORD)
        yag.send(EMAIL, subject, body)
    except Exception as e:
        print("Email failed:", e)

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
notify("Job Started", f"optimize_runscript.py started at {start_time}")

output_log = []

try:
    # -u is critical (forces unbuffered output)
    process = subprocess.Popen(
        ["python", "-u", "optimize_runscript.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output line-by-line
    for line in process.stdout:
        print(line, end="")           # echo to terminal
        output_log.append(line)        # save it for email

    process.wait()
    rc = process.returncode

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if rc == 0:
        notify(
            "Job SUCCESS",
            f"Completed at {end_time}\n\nFull Output:\n{''.join(output_log)}"
        )
    else:
        notify(
            "Job FAILED",
            f"Exited with code {rc} at {end_time}\n\nOutput:\n{''.join(output_log)}"
        )

except Exception as e:
    notify(
        "Job CRASHED",
        f"Exception:\n{traceback.format_exc()}\n\nOutput so far:\n{''.join(output_log)}"
    )
    raise
