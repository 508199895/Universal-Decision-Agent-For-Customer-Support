type TicketRequest = {
  userId?: unknown;
  message?: unknown;
};

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ detail: "请求内容必须是有效的 JSON。" }, { status: 400 });
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return Response.json({ detail: "请求内容必须是 JSON 对象。" }, { status: 400 });
  }

  const body = payload as TicketRequest;
  const userId = typeof body.userId === "string" ? body.userId.trim() : "";
  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!userId || !message) {
    return Response.json({ detail: "用户 ID 和消息内容不能为空。" }, { status: 400 });
  }

  const backendBaseUrl = (process.env.CUSTOMER_SUPPORT_API_URL ?? "http://127.0.0.1:8001").replace(/\/+$/, "");
  let endpoint: URL;
  try {
    endpoint = new URL(`${backendBaseUrl}/tickets`);
  } catch {
    return Response.json({ detail: "客服后端地址配置无效。" }, { status: 500 });
  }

  try {
    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ user_id: userId, message }),
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });

    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return Response.json(
      { detail: timedOut ? "连接客服后端超时，请确认工作流服务已启动。" : "无法连接客服后端，请确认工作流服务已启动。" },
      { status: timedOut ? 504 : 502 },
    );
  }
}
