# 客服对话前端

基于 Next.js App Router 的客服聊天前端。用户输入已有用户 ID 后提交消息，由 Next.js 服务端代理请求 FastAPI；前端通过 SSE 接收工单受理、状态更新和客服回复。

## 启动

先按仓库根目录 README 准备 `.env`、用户数据库、Qdrant 和飞书配置，并启动 FastAPI：

```powershell
.\.venv\python.exe -m uvicorn agentic.app.fastapi_server:app --host 127.0.0.1 --port 8001
```

然后在 `frontend/` 目录启动 Next.js：

```powershell
npm install
npm run dev
```

打开 <http://localhost:3000>，输入用户 ID 后进入聊天页。前端默认连接 `http://127.0.0.1:8001`。如 FastAPI 使用其他地址，在 `frontend/.env.local` 设置 `CUSTOMER_SUPPORT_API_URL`，然后重启 Next.js。

## 消息与事件

- 提交消息时，前端调用 `/api/tickets`，由 Next.js 代理到 FastAPI `POST /tickets`。
- 接受工单后，FastAPI 返回 `202 Accepted` 和工单编号；工作流在后台运行。
- 前端通过 `/api/events?userId=...` 连接 SSE 代理，订阅 `message.accepted`、`ticket.status`、`assistant.message`。
- SSE 当前是实时推送，不支持断线事件回放。事件落库失败时，后端仍会向当前在线订阅者发送原事件，但事件没有持久化序号。

登录仅记录用户 ID，不进行身份验证；后端要求该用户 ID 存在于用户数据库中。

## 检查

```powershell
npm run lint
npm run build
```
