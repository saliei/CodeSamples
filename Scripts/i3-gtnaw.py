#! /usr/bin/python
# go to next available empty workspace in i3 window manager.
# imagine you have occupied workspaces 1, 3, 4, 6,
# now you want to go to workspace 2(or 5) without pressing $mod+2
# bind this to a keyboard shortcut in ~/.i3/config like this:
# bindsym $mod+grave exec --no-startup-id <PATH-TO-THIS-SCRIPT>

import json
from subprocess import Popen, PIPE

# findout workspaces in use
process_get = Popen(["i3-msg", "-t", "get_workspaces"], stdout=PIPE, stderr=PIPE)
stdout, stderr = process_get.communicate()
# decode from bitestring and remove trailing newline
# and convert json output to a list of dictionaries
workspaces = json.loads(stdout.decode().rstrip())
# json output has a 'num' property which is the workspace number
workspace_nums = [workspace['num'] for workspace in workspaces]
workspace_range = range(1, workspace_nums[-1]+2)
available_workspaces = [x for x in workspace_range if x not in workspace_nums]
next_workspace = available_workspaces[0]

process_go = Popen(["i3-msg", "-t", "run_command", "workspace",
		   r"{}".format(next_workspace)], stdout=PIPE, stderr=PIPE)
stdout, stderr = process_go.communicate()
