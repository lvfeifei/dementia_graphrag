#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import csv
import os
import re
import sys
import json
from collections import defaultdict
from datetime import datetime
from neo4j import GraphDatabase

ROOT = Path(".")
SRC_DIR = ROOT / "data/v02_final_flat_for_import"
LOG_DIR = ROOT / "outputs/logs"
SNAPSHOT_COUNTS = SRC_DIR / "neo4j_counts_snapshot.csv"

ENTITY_FILES = {
    "Feature": "feature.csv",
    "Problem": "problem.csv",
    "Environment": "environment.csv",
}

ADVICE_FILES = {
    "ActivityAdvice": "activity_advice.csv",
    "EnvironmentAdvice": "environment_advice.csv",
    "ProblemAdvice": "problem_advice.csv",
}

RULE_SPECS = [
    {
        "file": "activity_advice_rule_processed.txt",
        "target_label": "ActivityAdvice",
        "combo_label": "ActivityCombinationNode",
        "single_rel": "single_to_activity_advice",
        "combo_rel": "comb_to_activity_advice",
        "combo_prefix": "LocalActivityComb",
    },
    {
        "file": "environment_advice_rule_processed.txt",
        "target_label": "EnvironmentAdvice",
        "combo_label": "EnvironmentCombinationNode",
        "single_rel": "single_to_environment_advice",
        "combo_rel": "comb_to_environment_advice",
        "combo_prefix": "LocalEnvironmentComb",
    },
    {
        "file": "problem_advice_rule_processed.txt",
        "target_label": "ProblemAdvice",
        "combo_label": "ProblemCombinationNode",
        "single_rel": "single_to_problem_advice",
        "combo_rel": "comb_to_problem_advice",
        "combo_prefix": "LocalProblemComb",
    },
    {
        "file": "problem_to_activity_advice_rule_processed.txt",
        "target_label": "ActivityAdvice",
        "combo_label": "Problem_To_ActivityCombinationNode",
        "single_rel": "single_to_problem_to_activity_advice",
        "combo_rel": "comb_to_problem_to_activity_advice",
        "combo_prefix": "LocalProblemToActivityComb",
    },
    {
        "file": "problem_to_environment_advice_rule_processed.txt",
        "target_label": "EnvironmentAdvice",
        "combo_label": "Problem_To_EnvironmentCombinationNode",
        "single_rel": "single_to_problem_to_environment_advice",
        "combo_rel": "comb_to_problem_to_environment_advice",
        "combo_prefix": "LocalProblemToEnvironmentComb",
    },
]

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


def qname(name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Unsafe Neo4j name: {name}")
    return f"`{name}`"


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_advice_content(row):
    name = (row.get("name") or "").strip()
    pics = (row.get("pics") or "").strip()
    pic_pos = (row.get("pic_pos") or "").strip()

    items = []
    if name:
        items.append({"type": "text", "con": name})

    if pics:
        pic_items = [p.strip() for p in re.split(r"[|；;]", pics) if p.strip()]
        if not pic_items:
            pic_items = [pics]

        for p in pic_items:
            lower = p.lower()
            item_type = "video" if ("视频" in p or ".mp4" in lower) else "pic"
            item = {"type": item_type, "con": p}
            if pic_pos:
                item["pic_pos"] = pic_pos
            items.append(item)

    if not items:
        items.append({"type": "text", "con": name})

    return str(items)


def parse_rule_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "->" not in line:
        raise ValueError(f"Bad rule line, missing -> : {line}")

    left, right = line.split("->", 1)
    left = left.strip()
    right = right.strip()

    if not right:
        raise ValueError(f"Bad rule line, empty target: {line}")

    cond_ids = []
    if left:
        for token in left.split(","):
            token = token.strip()
            if not token:
                continue
            cond_id = token.split(":", 1)[0].strip()
            if cond_id:
                cond_ids.append(cond_id)

    target_ids = [x.strip() for x in right.split(",") if x.strip()]
    return cond_ids, target_ids


class LocalKG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.id_to_label = {}
        self.advice_ids_by_label = defaultdict(set)

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

    def clear(self):
        print("⚠️ 清空本地 Neo4j 图谱：MATCH (n) DETACH DELETE n")
        self.run("MATCH (n) DETACH DELETE n")
        print("✅ 本地 Neo4j 已清空")

    def create_indexes(self):
        for label in LABELS:
            index_name = f"idx_{label}_id"
            self.run(
                f"CREATE INDEX {qname(index_name)} IF NOT EXISTS FOR (n:{qname(label)}) ON (n.id)"
            )
        print("✅ 索引已创建/确认")

    def merge_nodes(self, label, rows):
        self.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:{qname(label)} {{id: row.id}})
            SET n += row
            """,
            rows=rows,
        )
        print(f"✅ 导入节点 {label}: {len(rows)}")

    def import_entities(self):
        for label, filename in ENTITY_FILES.items():
            rows = []
            for r in read_csv_rows(SRC_DIR / filename):
                obj = {
                    "id": (r.get("id") or "").strip(),
                    "name": (r.get("name") or "").strip(),
                    "ptype": (r.get("ptype") or "").strip(),
                    "alias": (r.get("alias") or "").strip(),
                }
                if not obj["id"]:
                    raise ValueError(f"Empty id in {filename}: {r}")
                rows.append(obj)
                self.id_to_label[obj["id"]] = label
            self.merge_nodes(label, rows)

    def import_advices(self):
        for label, filename in ADVICE_FILES.items():
            rows = []
            for r in read_csv_rows(SRC_DIR / filename):
                obj = {}

                # 保留 CSV 中所有非空属性。
                # v02_final 新增 advice 节点会带有 source_id、source_title、evidence_text、heading_path、chunk_id 等追溯字段。
                for k, v in r.items():
                    if k is None:
                        continue
                    key = str(k).strip()
                    if not key:
                        continue
                    value = "" if v is None else str(v).strip()
                    if value != "":
                        obj[key] = value

                obj["id"] = (r.get("id") or "").strip()
                obj["name"] = (r.get("name") or "").strip()

                # 优先使用 Step36 已经生成好的 content。
                # 它已经符合服务器 v01 格式：[{'type': 'text', 'con': '……'}]
                raw_content = (r.get("content") or "").strip()
                if raw_content:
                    obj["content"] = raw_content
                else:
                    obj["content"] = make_advice_content(r)

                # 如果没有 content_text，则给一个纯文本兜底，方便后续 GraphRAG。
                if not obj.get("content_text"):
                    obj["content_text"] = obj["name"]

                if not obj["id"]:
                    raise ValueError(f"Empty id in {filename}: {r}")

                rows.append(obj)
                self.advice_ids_by_label[label].add(obj["id"])

            self.merge_nodes(label, rows)

    def set_scope_all(self, target_label, target_id):
        if target_id not in self.advice_ids_by_label[target_label]:
            raise ValueError(f"Unknown advice target: {target_label}:{target_id}")
        self.run(
            f"""
            MATCH (a:{qname(target_label)} {{id: $target_id}})
            SET a.scope = 'all'
            """,
            target_id=target_id,
        )

    def create_single_rel(self, cond_id, target_label, target_id, rel_type):
        source_label = self.id_to_label.get(cond_id)
        if not source_label:
            raise ValueError(f"Unknown condition id: {cond_id}")
        if target_id not in self.advice_ids_by_label[target_label]:
            raise ValueError(f"Unknown advice target: {target_label}:{target_id}")

        self.run(
            f"""
            MATCH (s:{qname(source_label)} {{id: $cond_id}})
            MATCH (a:{qname(target_label)} {{id: $target_id}})
            MERGE (s)-[:{qname(rel_type)}]->(a)
            """,
            cond_id=cond_id,
            target_id=target_id,
        )

    def create_combo_rel(self, cond_ids, target_label, target_id, combo_label, combo_id, combo_rel):
        if target_id not in self.advice_ids_by_label[target_label]:
            raise ValueError(f"Unknown advice target: {target_label}:{target_id}")

        self.run(
            f"""
            MERGE (c:{qname(combo_label)} {{id: $combo_id}})
            SET c.name = $combo_id
            """,
            combo_id=combo_id,
        )

        for cond_id in cond_ids:
            source_label = self.id_to_label.get(cond_id)
            if not source_label:
                raise ValueError(f"Unknown condition id: {cond_id}")

            self.run(
                f"""
                MATCH (s:{qname(source_label)} {{id: $cond_id}})
                MATCH (c:{qname(combo_label)} {{id: $combo_id}})
                MERGE (s)-[:{qname("part_of")}]->(c)
                """,
                cond_id=cond_id,
                combo_id=combo_id,
            )

        self.run(
            f"""
            MATCH (c:{qname(combo_label)} {{id: $combo_id}})
            MATCH (a:{qname(target_label)} {{id: $target_id}})
            MERGE (c)-[:{qname(combo_rel)}]->(a)
            """,
            combo_id=combo_id,
            target_id=target_id,
        )

    def import_rules(self):
        for spec in RULE_SPECS:
            path = SRC_DIR / spec["file"]
            single_count = 0
            combo_count = 0
            scope_count = 0
            line_count = 0

            for line in path.read_text(encoding="utf-8").splitlines():
                parsed = parse_rule_line(line)
                if parsed is None:
                    continue

                cond_ids, target_ids = parsed

                for target_id in target_ids:
                    line_count += 1

                    if len(cond_ids) == 0:
                        self.set_scope_all(spec["target_label"], target_id)
                        scope_count += 1

                    elif len(cond_ids) == 1:
                        self.create_single_rel(
                            cond_id=cond_ids[0],
                            target_label=spec["target_label"],
                            target_id=target_id,
                            rel_type=spec["single_rel"],
                        )
                        single_count += 1

                    else:
                        combo_count += 1
                        combo_id = f"{spec['combo_prefix']}{combo_count:04d}"
                        self.create_combo_rel(
                            cond_ids=cond_ids,
                            target_label=spec["target_label"],
                            target_id=target_id,
                            combo_label=spec["combo_label"],
                            combo_id=combo_id,
                            combo_rel=spec["combo_rel"],
                        )

            print(
                f"✅ 导入规则 {spec['file']}: "
                f"lines={line_count}, single={single_count}, combo={combo_count}, scope={scope_count}"
            )

    def load_expected_counts(self):
        expected = {}
        with SNAPSHOT_COUNTS.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                expected[(row["section"], row["name"])] = int(row["count"])
        return expected

    def validate(self):
        expected = self.load_expected_counts()
        actual = {}

        actual[("graph", "node_count")] = self.scalar("MATCH (n) RETURN count(n) AS c")
        actual[("graph", "relationship_count")] = self.scalar("MATCH ()-[r]->() RETURN count(r) AS c")

        for label in LABELS:
            actual[("label", label)] = self.scalar(f"MATCH (n:{qname(label)}) RETURN count(n) AS c")

        for rel in REL_TYPES:
            actual[("relationship", rel)] = self.scalar(f"MATCH ()-[r:{qname(rel)}]->() RETURN count(r) AS c")

        print("\n===== 本地 v02_final KG 重建验收 =====")
        mismatches = []

        for key in sorted(expected.keys()):
            exp = expected[key]
            act = actual.get(key, 0)
            mark = "✅" if exp == act else "❌"
            print(f"{mark} {key[0]:12s} {key[1]:45s} expected={exp:<5} actual={act}")
            if exp != act:
                mismatches.append({"section": key[0], "name": key[1], "expected": exp, "actual": act})

        duplicate_groups = self.scalar(
            """
            MATCH (a)-[r]->(b)
            WITH id(a) AS a_id, type(r) AS rel_type, id(b) AS b_id, count(*) AS c
            WHERE c > 1
            RETURN count(*) AS duplicate_groups
            """
        )

        scope_rows = self.run(
            """
            MATCH (n)
            WHERE n.scope = 'all'
            RETURN n.id AS id, labels(n) AS labels
            ORDER BY id
            """
        )
        scope_ids = sorted([r["id"] for r in scope_rows])

        print(f"\nDuplicate relationship groups: {duplicate_groups}")
        print("scope=all ids:", scope_ids)

        expected_scope = sorted(["HD14", "HD15", "J25", "J29"])

        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mismatches": mismatches,
            "duplicate_relationship_groups": duplicate_groups,
            "scope_all_ids": scope_ids,
            "expected_scope_all_ids": expected_scope,
            "passed": not mismatches and duplicate_groups == 0 and scope_ids == expected_scope,
        }

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = LOG_DIR / "step38_local_v02_final_kg_validation.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n📄 验收报告已写入：{out}")

        if report["passed"]:
            print("\n🎉 Local v02_final KG replication validation passed.")
            return True

        print("\n❌ Local v02_final KG replication validation failed.")
        return False


def check_required_files():
    required = [
        SRC_DIR / "feature.csv",
        SRC_DIR / "problem.csv",
        SRC_DIR / "environment.csv",
        SRC_DIR / "activity_advice.csv",
        SRC_DIR / "environment_advice.csv",
        SRC_DIR / "problem_advice.csv",
        SRC_DIR / "activity_advice_rule_processed.txt",
        SRC_DIR / "environment_advice_rule_processed.txt",
        SRC_DIR / "problem_advice_rule_processed.txt",
        SRC_DIR / "problem_to_activity_advice_rule_processed.txt",
        SRC_DIR / "problem_to_environment_advice_rule_processed.txt",
        SNAPSHOT_COUNTS,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_LOCAL_PASSWORD", ""))
    parser.add_argument("--confirm-local-reset", action="store_true")
    args = parser.parse_args()

    if "localhost" not in args.neo4j_uri and "127.0.0.1" not in args.neo4j_uri:
        raise RuntimeError(f"拒绝连接非本地 Neo4j: {args.neo4j_uri}")

    if not args.confirm_local_reset:
        raise RuntimeError("必须显式添加 --confirm-local-reset 才允许清空本地 Neo4j 并重建 v02_final")

    if not args.neo4j_password:
        raise RuntimeError("NEO4J_LOCAL_PASSWORD 为空，请先 export")

    check_required_files()

    kg = LocalKG(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        kg.clear()
        kg.create_indexes()
        kg.import_entities()
        kg.import_advices()
        kg.import_rules()
        ok = kg.validate()
        sys.exit(0 if ok else 2)
    finally:
        kg.close()


if __name__ == "__main__":
    main()
