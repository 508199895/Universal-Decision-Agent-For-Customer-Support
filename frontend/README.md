# 客服对话前端

基于 Next.js App Router 的客服聊天前端。登录只记录用户 ID，不进行身份验证。发送消息会通过 Next.js 服务端代理提交到 FastAPI `POST /tickets`，由后端校验用户记录并异步启动工作流。

## 启动

先在仓库根目录启动 FastAPI 工作流服务：

```powershell
.\.venv\python.exe -m uvicorn agentic.app.fastapi_server:app --host 127.0.0.1 --port 8001
```

然后在 `frontend/` 目录启动 Next.js：

```bash
npm install
npm run dev
```

打开 <http://localhost:3000>，输入用户 ID 后进入聊天页。

默认连接 `http://127.0.0.1:8001`，避免与已占用 8000 端口的其他服务冲突。如果 FastAPI 使用其他地址，在 `frontend/.env.local` 中设置 `CUSTOMER_SUPPORT_API_URL`（可参考 `.env.example`），然后重启 Next.js。

后端接口返回 `202 Accepted` 和工单编号后，前端会轮询工单状态；工作流结束时显示 AI 回复或具体失败状态。

## 检查

```bash
npm run lint
npm run build
```
