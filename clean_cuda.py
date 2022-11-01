import os
pid = list(set(os.popen('fuser -v /dev/nvidia*').read().split()))
pid = [p for p in pid if p.isdigit()]
kill_cmd = 'kill -9 ' + ' '.join(pid)
print(kill_cmd)
os.popen(kill_cmd)