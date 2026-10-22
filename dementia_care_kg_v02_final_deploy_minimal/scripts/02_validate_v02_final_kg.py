#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

from neo4j import GraphDatabase


ROOT = Path(".")
SRC_DIR = ROOT / "data/v02_final_flat_for_import"
OPS_CSV = SRC_DIR / "v02_final_update_operations.csv"
SNAPSHOT_COUNTS = SRC_DIR / "neo4j_counts_snapshot.csv"

LOG_DIR = ROOT / "outputs/logs"
REPORT_PATH = LOG_DIR / "step39_v02_final_kg_deep_validation.json"

DELETE_IDS = {
    "V02_UPD_0007",
    "V02_UPD_0024",
    "V02_UPD_0037",
    "V02_UPD_0041",
    "V02_UPD_0043",
    "V02_UPD_0045",
    "V02_UPD_0058",
    "V02_UPD_0059",
    "V02_UPD_0071",
    "V02_UPD_0079",
    "V02_UPD_0082",
    "V02_UPD_0084",
    "V02_UPD_0085",
    "V02_UPD_0087",
    "V02_UPD_0108",
    "V02_UPD_0110",
    "V02_UPD_0112",
    "V02_UPD_0122",
    "V02_UPD_0129",
}

LABELS = [
    "ActivityAdvice",
    "ActivityCombinationNode",
    "Environment",
    "EnvironmentAdvice",
    "EnvironmentCombinationNode",
    "Feature",
    "Problem",
    "ProblemAdvice",
    "ProblemCombinationNode",
    "Problem_To_ActivityCombinationNode",
    "Problem_To_EnvironmentCombinationNode",
]

REL_TYPES = [
    "comb_to_activity_advice",
    "comb_to_environment_advice",
    "comb_to_problem_advice",
    "comb_to_problem_to_activity_advice",
    "comb_to_problem_to_environment_advice",
    "part_of",
    "single_to_activity_advice",
    "single_to_environment_advice",
    "single_to_problem_advice",
    "single_to_problem_to_activity_advice",
    "single_to_problem_to_environment_advice",
]

REQUIRED_NEW_NODE_PROPS = [
    "id",
    "name",
    "content",
    "content_text",
    "update_id",
    "operation_id",
    "candidate_id",
    "source_id",
    "source_title",
    "source_type",
    "year",
    "evidence_text",
    "heading_path",
    "chunk_id",
    "condition_rule_text",
    "precheck_status",
]

EXPECTED_SCOPE_ALL = ["HD14", "HD15", "J25", "J29"]


def qname(name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Unsafe Neo4j name: {name}")
    return f"`{name}`"


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_counts(path: Path):
    rows = read_csv_rows(path)
    out = {}
    for r in rows:
        out[(r["section"], r["name"])] = int(r["count"])
    return out


def valid_v01_content_format(value):
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    try:
        obj = ast.literal_eval(s)
    except Exception:
        return False
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    if not isinstance(first, dict):
        return False
    return first.get("type") == "text" and "con" in first


class Validator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session() as session:
            return session.run(query, **params).data()

    def scalar(self, query, **params):
        rows = self.run(query, **params)
        if not rows:
            return 0
        return list(rows[0].values())[0]

    def node_props(self, label, node_id):
        rows = self.run(
            f"MATCH (n:{qname(label)} {{id: $id}}) RETURN properties(n) AS props",
            id=node_id,
        )
        if not rows:
            return None
        return rows[0]["props"]

    def rel_count(self, source_problem_id, rel_type, target_label, target_id):
        return self.scalar(
            f"""
            MATCH (p:Problem {{id: $pid}})-[r:{qname(rel_type)}]->(a:{qname(target_label)} {{id: $aid}})
            RETURN count(r) AS c
            """,
            pid=source_problem_id,
            aid=target_id,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_LOCAL_PASSWORD", ""))
    args = parser.parse_args()

    if "localhost" not in args.neo4j_uri and "127.0.0.1" not in args.neo4j_uri:
        raise RuntimeError(f"拒绝连接非本地 Neo4j: {args.neo4j_uri}")

    if not args.neo4j_password:
        raise RuntimeError("NEO4J_LOCAL_PASSWORD 为空，请先 export")

    if not OPS_CSV.exists():
        raise FileNotFoundError(f"缺少 operations 文件：{OPS_CSV}")
    if not SNAPSHOT_COUNTS.exists():
        raise FileNotFoundError(f"缺少 counts snapshot：{SNAPSHOT_COUNTS}")

    operations = read_csv_rows(OPS_CSV)
    expected_counts = read_counts(SNAPSHOT_COUNTS)

    errors = []
    warnings = []

    v = Validator(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        # 1. 总 count 验证
        actual_counts = {}
        actual_counts[("graph", "node_count")] = v.scalar("MATCH (n) RETURN count(n) AS c")
        actual_counts[("graph", "relationship_count")] = v.scalar("MATCH ()-[r]->() RETURN count(r) AS c")

        for label in LABELS:
            actual_counts[("label", label)] = v.scalar(f"MATCH (n:{qname(label)}) RETURN count(n) AS c")

        for rel in REL_TYPES:
            actual_counts[("relationship", rel)] = v.scalar(f"MATCH ()-[r:{qname(rel)}]->() RETURN count(r) AS c")

        count_mismatches = []
        for key, exp in expected_counts.items():
            act = actual_counts.get(key, 0)
            if act != exp:
                count_mismatches.append({
                    "section": key[0],
                    "name": key[1],
                    "expected": exp,
                    "actual": act,
                })

        if count_mismatches:
            errors.append({"count_mismatches": count_mismatches})

        # 2. 120 条操作逐条查 Neo4j 节点和关系
        if len(operations) != 120:
            errors.append(f"operations 应为 120 条，实际={len(operations)}")

        missing_nodes = []
        missing_relationships = []
        missing_props = []
        bad_content_format = []
        bad_content_text = []
        update_id_mismatch = []
        evidence_suspicious = []

        for op in operations:
            operation_id = op.get("operation_id", "")
            update_id = op.get("update_id", "")
            label = op.get("target_label", "")
            aid = op.get("generated_advice_id", "")
            pid = op.get("target_problem_id_single", "")
            rel = op.get("relationship", "")

            props = v.node_props(label, aid)
            if props is None:
                missing_nodes.append({
                    "operation_id": operation_id,
                    "label": label,
                    "id": aid,
                })
                continue

            # 必要属性
            for key in REQUIRED_NEW_NODE_PROPS:
                value = props.get(key)
                if value is None or str(value).strip() == "":
                    missing_props.append({
                        "operation_id": operation_id,
                        "label": label,
                        "id": aid,
                        "missing_prop": key,
                    })

            # update_id / operation_id 对齐
            if str(props.get("update_id", "")).strip() != update_id:
                update_id_mismatch.append({
                    "operation_id": operation_id,
                    "id": aid,
                    "expected_update_id": update_id,
                    "actual_update_id": props.get("update_id"),
                })

            if str(props.get("operation_id", "")).strip() != operation_id:
                update_id_mismatch.append({
                    "operation_id": operation_id,
                    "id": aid,
                    "expected_operation_id": operation_id,
                    "actual_operation_id": props.get("operation_id"),
                })

            # content 格式
            if not valid_v01_content_format(props.get("content")):
                bad_content_format.append({
                    "operation_id": operation_id,
                    "label": label,
                    "id": aid,
                    "content_head": str(props.get("content", ""))[:100],
                })

            if str(props.get("content_text", "")).strip() == "":
                bad_content_text.append({
                    "operation_id": operation_id,
                    "label": label,
                    "id": aid,
                })

            # 证据字段粗查
            for key in ["evidence_text", "heading_path", "chunk_id", "source_id", "source_title"]:
                val = str(props.get(key, "")).strip()
                if len(val) < 2:
                    evidence_suspicious.append({
                        "operation_id": operation_id,
                        "id": aid,
                        "field": key,
                        "value": val,
                    })

            # 关系检查
            rc = v.rel_count(pid, rel, label, aid)
            if rc != 1:
                missing_relationships.append({
                    "operation_id": operation_id,
                    "source_problem_id": pid,
                    "relationship": rel,
                    "target_label": label,
                    "target_advice_id": aid,
                    "actual_count": rc,
                })

        if missing_nodes:
            errors.append({"missing_new_nodes": missing_nodes[:50]})
        if missing_relationships:
            errors.append({"missing_or_duplicate_new_relationships": missing_relationships[:50]})
        if missing_props:
            errors.append({"missing_new_node_props": missing_props[:80]})
        if bad_content_format:
            errors.append({"bad_content_format": bad_content_format[:50]})
        if bad_content_text:
            errors.append({"bad_content_text": bad_content_text[:50]})
        if update_id_mismatch:
            errors.append({"update_or_operation_id_mismatch": update_id_mismatch[:50]})
        if evidence_suspicious:
            errors.append({"evidence_suspicious": evidence_suspicious[:50]})

        # 3. 删除的 19 条绝不能在 Neo4j 作为新增节点 update_id 出现
        leaked_deleted = []
        for uid in sorted(DELETE_IDS):
            c = v.scalar(
                """
                MATCH (n)
                WHERE n.update_id = $uid
                RETURN count(n) AS c
                """,
                uid=uid,
            )
            if c:
                leaked_deleted.append({"update_id": uid, "node_count": c})
        if leaked_deleted:
            errors.append({"deleted_update_ids_found_in_neo4j": leaked_deleted})

        # 4. V02_UPD_0075 拆分检查
        split_0075 = v.run(
            """
            MATCH (n)
            WHERE n.update_id = "V02_UPD_0075"
            RETURN labels(n) AS labels, n.id AS id, n.operation_id AS operation_id,
                   n.target_problem_id AS target_problem_id, n.original_new_advice_id AS original_new_advice_id
            ORDER BY n.operation_id
            """
        )
        if len(split_0075) != 2:
            errors.append({"V02_UPD_0075_split_node_count_expected_2": split_0075})

        split_targets = sorted([r.get("target_problem_id") for r in split_0075])
        if split_targets != ["H14", "H2"]:
            errors.append({"V02_UPD_0075_split_targets_wrong": split_0075})

        # 5. H2-10 自动重编号检查
        h2_renamed = v.run(
            """
            MATCH (n)
            WHERE n.update_id = "V02_UPD_0128"
            RETURN labels(n) AS labels, n.id AS id, n.original_new_advice_id AS original_new_advice_id,
                   n.id_conflict_auto_renamed AS id_conflict_auto_renamed,
                   n.conflicting_new_advice_id AS conflicting_new_advice_id,
                   n.target_problem_id AS target_problem_id
            """
        )
        if len(h2_renamed) != 1:
            errors.append({"V02_UPD_0128_node_count_expected_1": h2_renamed})
        else:
            r = h2_renamed[0]
            if r.get("id") != "H2-11":
                errors.append({"V02_UPD_0128_expected_generated_id_H2_11": r})
            if r.get("original_new_advice_id") != "H2-10":
                errors.append({"V02_UPD_0128_original_id_not_preserved": r})
            if r.get("id_conflict_auto_renamed") != "TRUE":
                errors.append({"V02_UPD_0128_rename_flag_wrong": r})

        # 6. 重复关系
        duplicate_relationship_groups = v.scalar(
            """
            MATCH (a)-[r]->(b)
            WITH id(a) AS a_id, type(r) AS rel_type, id(b) AS b_id, count(*) AS c
            WHERE c > 1
            RETURN count(*) AS duplicate_groups
            """
        )
        if duplicate_relationship_groups != 0:
            errors.append({"duplicate_relationship_groups": duplicate_relationship_groups})

        # 7. scope=all 检查
        scope_rows = v.run(
            """
            MATCH (n)
            WHERE n.scope = "all"
            RETURN n.id AS id, labels(n) AS labels
            ORDER BY id
            """
        )
        scope_ids = sorted([r["id"] for r in scope_rows])
        if scope_ids != EXPECTED_SCOPE_ALL:
            errors.append({"scope_all_ids_wrong": scope_ids})

        # 8. 新增节点按 source 分布
        source_dist = v.run(
            """
            MATCH (n)
            WHERE n.update_id STARTS WITH "V02_UPD_"
            RETURN n.source_id AS source_id, n.source_title AS source_title, count(n) AS count
            ORDER BY source_id
            """
        )

        label_dist = v.run(
            """
            MATCH (n)
            WHERE n.update_id STARTS WITH "V02_UPD_"
            RETURN labels(n) AS labels, count(n) AS count
            ORDER BY count DESC
            """
        )

        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "PASS" if not errors else "FAIL",
            "inputs": {
                "ops_csv": str(OPS_CSV),
                "snapshot_counts": str(SNAPSHOT_COUNTS),
            },
            "summary": {
                "operation_count": len(operations),
                "count_mismatches": count_mismatches,
                "missing_new_nodes_count": len(missing_nodes),
                "missing_relationships_count": len(missing_relationships),
                "missing_props_count": len(missing_props),
                "bad_content_format_count": len(bad_content_format),
                "deleted_update_ids_leaked_count": len(leaked_deleted),
                "duplicate_relationship_groups": duplicate_relationship_groups,
                "scope_all_ids": scope_ids,
                "V02_UPD_0075_split_nodes": split_0075,
                "V02_UPD_0128_renamed_node": h2_renamed,
                "source_distribution": source_dist,
                "label_distribution": label_dist,
                "actual_counts": {f"{k[0]}:{k[1]}": v for k, v in actual_counts.items()},
            },
            "errors": errors,
            "warnings": warnings,
        }

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("===== STEP39 V02_FINAL KG DEEP VALIDATION =====")
        print("operation_count:", len(operations))
        print("missing_new_nodes:", len(missing_nodes))
        print("missing_relationships:", len(missing_relationships))
        print("missing_props:", len(missing_props))
        print("bad_content_format:", len(bad_content_format))
        print("deleted_update_ids_leaked:", len(leaked_deleted))
        print("duplicate_relationship_groups:", duplicate_relationship_groups)
        print("scope_all_ids:", scope_ids)
        print("V02_UPD_0075_split_nodes:", split_0075)
        print("V02_UPD_0128_renamed_node:", h2_renamed)
        print("report:", REPORT_PATH)
        print("STATUS =", report["status"])

        if errors:
            print("\nERRORS:")
            for e in errors:
                print("-", e)
            raise SystemExit(1)

    finally:
        v.close()


if __name__ == "__main__":
    main()
