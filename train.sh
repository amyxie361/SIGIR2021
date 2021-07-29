python -m src.train \
    --triples /data/y247xie/00_data/MSMARCO/triples.train.small.expanded.tsv \
    --maxsteps 100000 \
    --bsize 32 \
    --accum 2 \
    --output_dir output.train \
    --similarity cosine \
    --dim 128 \
    --query_maxlen 32 \
    --doc_maxlen 180


