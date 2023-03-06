until python pretrain.py --generate True --lang=PROP --seed=121; do
    echo "generator crashed with exit code $?.  Respawning.." >&2
    sleep 1
done