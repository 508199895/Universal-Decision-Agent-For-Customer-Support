版本：v2026-10-31-before-collection-default
日期：2026-10-31
原因：记录 Qdrant MCP 在缺少 COLLECTION_NAME 时，将 collection_name 暴露为模型必填参数的修改前状态。
当前状态：项目根目录 .env 不存在；进程环境未设置 COLLECTION_NAME；.env.example 中的目标集合为 uda-hub-retrieval-eval。
对应生成文件：tool_schema.json，记录修改前 qdrant-find 的实际参数 schema。
