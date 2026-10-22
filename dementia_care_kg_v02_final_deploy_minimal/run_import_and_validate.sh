#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"

if [ -z "${NEO4J_LOCAL_PASSWORD:-}" ]; then
  echo "ERROR: NEO4J_LOCAL_PASSWORD is empty."
  echo "Please export NEO4J_LOCAL_PASSWORD first."
  exit 1
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "NEO4J_URI=${NEO4J_URI}"
echo "NEO4J_USER=${NEO4J_USER}"
echo ""
echo "This script will clear the target Neo4j database and import v02_final."
echo "Only run it on a new/test/separate Neo4j instance."
echo ""

"${PYTHON_BIN}" scripts/01_import_v02_final_to_neo4j.py \
  --neo4j-uri "${NEO4J_URI}" \
  --neo4j-user "${NEO4J_USER}" \
  --confirm-local-reset

"${PYTHON_BIN}" scripts/02_validate_v02_final_kg.py \
  --neo4j-uri "${NEO4J_URI}" \
  --neo4j-user "${NEO4J_USER}"

echo ""
echo "v02_final KG import and validation finished."
