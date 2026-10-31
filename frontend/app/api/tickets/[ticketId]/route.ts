type RouteContext = {
  params: Promise<{ ticketId: string }>;
};

export async function GET(request: Request, { params }: RouteContext) {
  const userId = new URL(request.url).searchParams.get("userId")?.trim();
  if (!userId) {
    return Response.json({ detail: "缺少用户 ID。" }, { status: 400 });
  }

  const { ticketId } = await params;
  const backendBaseUrl = (process.env.CUSTOMER_SUPPORT_API_URL ?? "http://127.0.0.1:8001").replace(/\/+$/, "");
  let endpoint: URL;
  try {
    endpoint = new URL(`${backendBaseUrl}/tickets/${encodeURIComponent(ticketId)}`);
    endpoint.searchParams.set("user_id", userId);
  } catch {
    return Response.json({ detail: "客服后端地址配置无效。" }, { status: 500 });
  }

  try {
    const upstream = await fetch(endpoint, { cache: "no-store", signal: AbortSignal.timeout(10_000) });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return Response.json(
      { detail: timedOut ? "查询工单结果超时。" : "无法连接客服后端。" },
      { status: timedOut ? 504 : 502 },
    );
  }
}
