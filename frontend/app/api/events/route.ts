export async function GET(request: Request) {
  const userId = new URL(request.url).searchParams.get("userId")?.trim();
  if (!userId) {
    return Response.json({ detail: "缺少用户 ID。" }, { status: 400 });
  }

  const backendBaseUrl = (process.env.CUSTOMER_SUPPORT_API_URL ?? "http://127.0.0.1:8001").replace(/\/+$/, "");
  let endpoint: URL;
  try {
    endpoint = new URL(`${backendBaseUrl}/events`);
    endpoint.searchParams.set("user_id", userId);
  } catch {
    return Response.json({ detail: "客服后端地址配置无效。" }, { status: 500 });
  }

  try {
    const upstream = await fetch(endpoint, {
      headers: { Accept: "text/event-stream" },
      cache: "no-store",
      signal: request.signal,
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return Response.json({ detail: "无法连接客服后端。" }, { status: 502 });
  }
}
