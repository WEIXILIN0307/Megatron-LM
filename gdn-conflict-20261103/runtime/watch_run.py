"""Run an existing test launcher with timed traceback dumps for diagnosing waits."""
import faulthandler
import runpy
import sys

faulthandler.dump_traceback_later(120, repeat=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
