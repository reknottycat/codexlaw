#!/usr/bin/env bash
set -euo pipefail

mkdir -p vendor
clone() { test -d "vendor/$2/.git" || git clone --depth 1 "$1" "vendor/$2"; }
clone https://github.com/WenzhuoXu/lawgent.git lawgent
clone https://github.com/openai/codex.git codex
clone https://github.com/neo4j-product-examples/graphrag-contract-review.git graphrag-contract-review
clone https://github.com/Fan-Luo/Legal-RAG.git legal-rag
echo "Upstream sources ready under vendor/. Review each repository license before redistributing data."
