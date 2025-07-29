#!/bin/bash

# Script per eseguire 03_embedding_multi.py su vari modelli SentenceTransformer
# Funziona su macOS

echo ">>> Starting batch execution..."

MODELS=(
  "all-mpnet-base-v2"
  "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
  "intfloat/e5-base-v2"
  "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
  "multi-qa-mpnet-base-dot-v1"
  "nli-roberta-base-v2"
  "princeton-nlp/sup-simcse-roberta-base"
  "sentence-t5-base"
)

SCRIPT="02_embedding_multi.py"

for MODEL in "${MODELS[@]}"
do
  echo ""
  echo ">>> Running model: $MODEL"
  python3 "$SCRIPT" --model "$MODEL"
done

echo ""
echo ">>> All models executed."