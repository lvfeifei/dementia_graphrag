# GitHub 手动上传步骤

目标：把第二套 GraphRAG Web 服务上传到仓库 `lvfeifei/dementia_graphrag` 的 `dev` 分支。

## 1. 先确认有写权限

打开：

https://github.com/lvfeifei/dementia_graphrag

确认右上角登录的是 `lvfeifei` 或其他有写权限的账号。公开仓库能直接打开，不代表有写权限。

## 2. 创建 dev 分支

1. 点击仓库左侧分支下拉框，默认通常是 `main`。
2. 在搜索框输入 `dev`。
3. 如果没有 `dev`，点击 **Create branch: dev from main**。
4. 切换到 `dev` 分支。

## 3. 上传文件夹

1. 解压 `dementia_graphrag_second_service_manual_upload.zip`。
2. 进入解压后的目录。
3. 在 GitHub 仓库页面确认当前分支是 `dev`。
4. 点击 **Add file**。
5. 点击 **Upload files**。
6. 把解压出来的整个文件夹 `dementia_care_graphrag_second_service` 拖到网页上传区域。
7. 等待页面列出所有文件，确认包含：
   - `dementia_care_graphrag_second_service/api_server.py`
   - `dementia_care_graphrag_second_service/web/index.html`
   - `dementia_care_graphrag_second_service/web/app.js`
   - `dementia_care_graphrag_second_service/web/styles.css`
   - `dementia_care_graphrag_second_service/SECOND_SERVICE_DEPLOY.md`
8. 页面下方填写提交信息：

Add independent second GraphRAG web service

9. 选择 **Commit directly to the dev branch**。
10. 点击 **Commit changes**。

## 4. 上传后检查

在 `dev` 分支打开：

https://github.com/lvfeifei/dementia_graphrag/tree/dev/dementia_care_graphrag_second_service

确认 `web` 文件夹存在，根目录有 `api_server.py`。

## 说明

- 这个压缩包不包含 `.env`、SSH 私钥、真实密码、日志或运行态文件。
- 包含示例环境文件：`.env.example`、`.env.deploy.example`。
- 服务真实部署仍使用服务器上的 `.env`，不会因为上传 GitHub 自动修改线上服务。
- 上传 GitHub 后，还需要在服务器拉取代码并重启服务，网页效果才会更新。
