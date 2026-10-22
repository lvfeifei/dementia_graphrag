# v02_final 知识图谱部署包使用说明

## 1. 部署包用途

本部署包用于在服务器或测试环境中部署 `v02_final` 失智老年人照护知识图谱。

部署包目录：

    dementia_care_kg_v02_final_deploy_minimal/

压缩包文件：

    dementia_care_kg_v02_final_deploy_minimal.tar.gz

本部署包包含：

    1. v02_final 知识图谱导入数据
    2. Neo4j 导入脚本
    3. Neo4j 验收脚本
    4. 一键导入与验收脚本
    5. 运行所需 Python 依赖文件
    6. 部署包清单文件

本部署包不包含：

    1. Neo4j 安装包
    2. GraphRAG 代码
    3. 前端或小程序代码
    4. 服务器密码
    5. Neo4j 密码
    6. API Key

---

## 2. 图谱版本与预期规模

当前知识图谱版本：

    v02_final

导入成功后，预期 Neo4j 图谱规模如下：

    node_count = 1087
    relationship_count = 2008
    ProblemAdvice = 396
    ActivityAdvice = 52
    EnvironmentAdvice = 45

节点标签包括：

    ActivityAdvice
    ActivityCombinationNode
    Environment
    EnvironmentAdvice
    EnvironmentCombinationNode
    Feature
    Problem
    ProblemAdvice
    ProblemCombinationNode
    Problem_To_ActivityCombinationNode
    Problem_To_EnvironmentCombinationNode

关系类型包括：

    comb_to_activity_advice
    comb_to_environment_advice
    comb_to_problem_advice
    comb_to_problem_to_activity_advice
    part_of
    single_to_activity_advice
    single_to_environment_advice
    single_to_problem_advice
    single_to_problem_to_activity_advice
    single_to_problem_to_environment_advice

说明：

    comb_to_problem_to_environment_advice 在当前图谱中数量为 0。
    因此如果导出文件或 Neo4j 关系类型列表中没有该关系类型，属于正常情况。

---

## 3. 部署包结构

部署包结构如下：

    dementia_care_kg_v02_final_deploy_minimal/
    ├── PACKAGE_MANIFEST.json
    ├── README_DEPLOY.md
    ├── data/
    │   ├── v02_final_asset_manifest.json
    │   └── v02_final_flat_for_import/
    │       ├── activity_advice.csv
    │       ├── activity_advice_rule_processed.txt
    │       ├── environment.csv
    │       ├── environment_advice.csv
    │       ├── environment_advice_rule_processed.txt
    │       ├── feature.csv
    │       ├── neo4j_counts_snapshot.csv
    │       ├── problem.csv
    │       ├── problem_advice.csv
    │       ├── problem_advice_rule_processed.txt
    │       ├── problem_to_activity_advice_rule_processed.txt
    │       ├── problem_to_environment_advice_rule_processed.txt
    │       └── v02_final_update_operations.csv
    ├── outputs/
    │   └── logs/
    ├── requirements.txt
    ├── run_import_and_validate.sh
    └── scripts/
        ├── 01_import_v02_final_to_neo4j.py
        └── 02_validate_v02_final_kg.py

核心文件说明：

| 文件或文件夹 | 说明 |
|---|---|
| data/v02_final_flat_for_import/ | v02_final 知识图谱导入数据 |
| data/v02_final_asset_manifest.json | v02_final 资产清单 |
| scripts/01_import_v02_final_to_neo4j.py | 将 v02_final 导入 Neo4j |
| scripts/02_validate_v02_final_kg.py | 验收 Neo4j 中的 v02_final 图谱 |
| run_import_and_validate.sh | 一键导入并验收 |
| outputs/logs/ | 导入和验收日志输出目录 |
| requirements.txt | Python 依赖 |
| PACKAGE_MANIFEST.json | 部署包清单 |

---

## 4. 重要提醒：Neo4j Community 版本通常只有一个默认数据库

如果部署环境使用的是：

    neo4j-community-4.4.9

需要注意：

    Neo4j Community 4.4.9 通常只有一个默认数据库。
    如果当前 Neo4j 中已经有 v01 图谱，直接运行本部署包的导入脚本，会清空当前数据库并导入 v02_final。

因此，强烈建议部署人员采用下面两种方式之一。

### 方案 A：单独启动一个 v02_final Neo4j 实例

这是推荐方案。

含义是：

    原有 v01 Neo4j 继续保留；
    v02_final 使用另一个 Neo4j 目录和另一组端口；
    后续 GraphRAG 或后端服务连接 v02_final 的 Bolt 端口。

示例：

    v01 Neo4j:
        Browser: 7474
        Bolt: 7687

    v02_final Neo4j:
        Browser: 7475
        Bolt: 7688

这样 v01 和 v02_final 可以同时存在，互不影响。

### 方案 B：覆盖原有 Neo4j

不推荐直接采用该方案。

如果必须覆盖原有 Neo4j，部署人员必须先确认：

    1. 原有 v01 图谱已经完整备份；
    2. 依赖原 v01 图谱的后端服务已经停止；
    3. 当前数据库可以被清空；
    4. 导入 v02_final 后，原 v01 图谱不再需要立刻恢复。

只有在确认以上条件后，才可以在原 Neo4j 实例上运行导入脚本。

---

## 5. 推荐部署方式：单独 Neo4j 实例

以下以 Neo4j Community 4.4.9 为例。

假设服务器已有 Neo4j 目录：

    /root/software/neo4j-community-4.4.9

可以复制一份给 v02_final：

    cp -R /root/software/neo4j-community-4.4.9 /root/software/neo4j-community-4.4.9-v02

进入 v02_final 的 Neo4j 目录：

    cd /root/software/neo4j-community-4.4.9-v02

编辑配置：

    vim conf/neo4j.conf

如果原有 v01 使用 7474 和 7687，则建议将 v02_final 改为：

    dbms.connector.bolt.listen_address=:7688
    dbms.connector.http.listen_address=:7475

如果配置文件中还有 advertised_address，也需要对应调整，例如：

    dbms.connector.bolt.advertised_address=:7688
    dbms.connector.http.advertised_address=:7475

启动 v02_final Neo4j：

    cd /root/software/neo4j-community-4.4.9-v02
    bin/neo4j start

查看状态：

    bin/neo4j status

浏览器访问地址：

    http://服务器IP:7475

Bolt 连接地址：

    bolt://127.0.0.1:7688

---

## 6. 上传并解压部署包

将部署包上传到服务器后，执行：

    tar -xzf dementia_care_kg_v02_final_deploy_minimal.tar.gz
    cd dementia_care_kg_v02_final_deploy_minimal

进入目录后，可以检查文件：

    ls
    tree -L 3

---

## 7. 安装 Python 依赖

如果服务器 Python 是：

    /usr/local/python3.9/bin/python3.9

建议执行：

    export PYTHON_BIN=/usr/local/python3.9/bin/python3.9
    ${PYTHON_BIN} -m pip install -r requirements.txt

如果服务器默认 python3 可用，也可以执行：

    python3 -m pip install -r requirements.txt

本部署包主要依赖：

    neo4j

---

## 8. 设置 Neo4j 连接信息

如果采用推荐方案，v02_final Neo4j Bolt 端口为 7688，则执行：

    export NEO4J_URI=bolt://127.0.0.1:7688
    export NEO4J_USER=neo4j
    export NEO4J_LOCAL_PASSWORD='请替换为真实Neo4j密码'

如果使用默认 Bolt 端口 7687，则执行：

    export NEO4J_URI=bolt://127.0.0.1:7687
    export NEO4J_USER=neo4j
    export NEO4J_LOCAL_PASSWORD='请替换为真实Neo4j密码'

如果需要指定 Python：

    export PYTHON_BIN=/usr/local/python3.9/bin/python3.9

注意：

    不要把真实 Neo4j 密码写进 README、代码、Git 仓库或公开文档。
    密码只应通过环境变量传入。

---

## 9. 一键导入并验收

确认目标 Neo4j 是 v02_final 的单独实例后，执行：

    bash run_import_and_validate.sh

该脚本会依次执行：

    1. scripts/01_import_v02_final_to_neo4j.py
    2. scripts/02_validate_v02_final_kg.py

注意：

    导入脚本会清空目标 Neo4j 当前数据库。
    请只在新的、测试的、单独的 v02_final Neo4j 实例上运行。
    不要在仍需保留 v01 的 Neo4j 实例上直接运行。

---

## 10. 成功标志

导入成功后，应看到类似结果：

    Local v02_final KG replication validation passed.

验收成功后，应看到：

    STATUS = PASS

关键验收指标应为：

    node_count = 1087
    relationship_count = 2008
    missing_new_nodes = 0
    missing_relationships = 0
    missing_props = 0
    bad_content_format = 0
    duplicate_relationship_groups = 0
    STATUS = PASS

如果以上指标通过，说明 v02_final 知识图谱部署成功。

---

## 11. 单独运行导入脚本

如果不使用一键脚本，可以单独运行导入：

    ${PYTHON_BIN:-python3} scripts/01_import_v02_final_to_neo4j.py \
      --neo4j-uri "${NEO4J_URI}" \
      --neo4j-user "${NEO4J_USER}" \
      --confirm-local-reset

说明：

    --confirm-local-reset 是必要参数。
    它用于提醒部署人员：该脚本会清空目标 Neo4j 并重新导入 v02_final。

---

## 12. 单独运行验收脚本

导入后，可以单独运行验收：

    ${PYTHON_BIN:-python3} scripts/02_validate_v02_final_kg.py \
      --neo4j-uri "${NEO4J_URI}" \
      --neo4j-user "${NEO4J_USER}"

验收报告会写入：

    outputs/logs/

---

## 13. 验收日志位置

导入和验收结束后，检查：

    ls -lh outputs/logs

常见日志包括：

    step38_local_v02_final_kg_validation.json
    step39_v02_final_kg_deep_validation.json

这些文件用于确认实际导入结果是否符合 v02_final 预期规模。

---

## 14. 如果出现连接失败

如果报错显示无法连接 Neo4j，请部署人员检查：

    1. Neo4j 是否已经启动；
    2. Bolt 端口是否正确；
    3. NEO4J_URI 是否正确；
    4. Neo4j 用户名是否正确；
    5. NEO4J_LOCAL_PASSWORD 是否正确；
    6. 防火墙或服务器安全组是否阻断端口；
    7. v02_final Neo4j 是否与 v01 Neo4j 使用了冲突端口。

查看 Neo4j 状态：

    bin/neo4j status

查看端口占用：

    lsof -i:7688
    lsof -i:7475

如果使用默认端口：

    lsof -i:7687
    lsof -i:7474

---

## 15. 如果出现密码错误

如果 Neo4j 密码错误，通常会出现认证失败。

请重新设置：

    export NEO4J_LOCAL_PASSWORD='正确密码'

然后重新运行：

    bash run_import_and_validate.sh

---

## 16. 如果必须覆盖原有 v01 图谱

不推荐，但如果部署人员确认必须覆盖，应先做备份。

建议先停止依赖原图谱的服务，然后停止 Neo4j：

    cd /root/software/neo4j-community-4.4.9
    bin/neo4j stop

备份原数据目录：

    tar -czf neo4j_v01_data_backup_$(date +%Y%m%d_%H%M%S).tar.gz data/

再启动 Neo4j：

    bin/neo4j start

确认可以覆盖后，再进入本部署包目录运行：

    bash run_import_and_validate.sh

---

## 17. 与后续 GraphRAG 的关系

本包只负责部署 v02_final 知识图谱。

后续 GraphRAG 或后端服务应通过环境变量连接该 Neo4j，例如：

    export NEO4J_URI=bolt://127.0.0.1:7688
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD='真实Neo4j密码'

推荐调用架构：

    前端或小程序
        ↓
    GraphRAG 后端 API
        ↓
    Neo4j v02_final
        ↓
    大语言模型、PDF 工具、ICS 日历工具

注意：

    前端或小程序不应直接连接 Neo4j。
    Neo4j 密码不应暴露到前端。
    GraphRAG 后端负责检索、生成、证据追溯和安全提醒。

---

## 18. Neo4j Aura 情况

如果后续使用 Neo4j Aura，需要单独确认：

    1. Aura 的 URI；
    2. Aura 的用户名；
    3. Aura 的密码；
    4. 是否使用 neo4j+s:// 协议；
    5. 是否允许导入脚本清空并批量写入；
    6. 是否需要修改脚本中的本地安全限制。

当前部署包主要面向服务器上的 Neo4j Community 4.4.9。

如果使用 Neo4j Aura，建议部署人员先单独创建 Aura 测试库，再根据 Aura 连接方式调整脚本。

---

## 19. 最终部署成功标准

部署成功需要同时满足：

    1. 导入脚本运行完成；
    2. 验收脚本显示 STATUS = PASS；
    3. node_count = 1087；
    4. relationship_count = 2008；
    5. outputs/logs 中生成验收报告；
    6. GraphRAG 或后端服务可以通过 Bolt URI 连接该 v02_final 图谱。

满足以上条件后，可认为 v02_final 知识图谱部署完成。
