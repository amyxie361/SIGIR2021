# SIGIR2021


## Train

```
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
```
--triples: 
query_text \t pos_doc \t neg_doc, all docs are expanded by docT5query

## Index
``` 
python -m src.index \
    --output_path ./collections/origin_reindex \
    --collection /home/y247xie/00_data/MSMARCO/collection.tsv \
    --ckpt /home/y247xie/01_exps/DeepImpact/official/colbert-12layers-100000.dnn
```
--collection:
expanded collection tsv
pid \t doc

Training triples:
https://drive.google.com/file/d/1SlVfdqdtAjbf7T0tnaZ7As3esLf9s1TZ/view?usp=sharing

Expanded collection:
https://drive.google.com/file/d/10PKQeTsfxQclVlQs6dYDPyg6vaMR4cUX/view?usp=sharing

Model checkpoint:
https://drive.google.com/file/d/1WQJcgWI5NRNQz8aNrWFx72SqSaM2XskH/view?usp=sharing

