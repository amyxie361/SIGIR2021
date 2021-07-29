export CKPT=32000
for i in `seq 1 89`; do 
python ./quantize.py \
    --input indexes/collection_initial_expand/origin_docs/ckpt_${CKPT}_${i} \
    --output indexes/collection_initial_expand/origin_docs_${CKPT}/docs${i}.json 
done

