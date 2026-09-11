#!/bin/bash
# Poll the A1 W3 queue on b3 every 300 s and print a one-line state per poll.
while true; do
  ssh b3 'q=$(grep -c . /home/alex/runs/w3a11_b3_queue.txt 2>/dev/null || echo 0)
    r=$(pgrep -fc "[d]scc_train.py" || echo 0)
    d=$(ls -d /home/alex/runs/dscc/dscc_w3a11_*/held_final.json 2>/dev/null | wc -l)
    e=$(for f in /home/alex/runs/dscc/dscc_w3a11_*/history.json; do [ -f "$f" ] && python3 -c "import json,sys;h=json.load(open(sys.argv[1]));print(h[-1][\"epoch\"] if h else -1)" "$f"; done | tr "\n" " ")
    echo "$(date +%F\ %T) queued $q running $r done $d epochs: $e"'
  sleep 300
done
