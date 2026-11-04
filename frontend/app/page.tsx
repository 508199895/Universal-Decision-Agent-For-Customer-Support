"use client";

import { FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

const terminalStatuses = new Set(["success", "error", "timeout", "interrupted"]);
const statusLabels: Record<string, string> = {
  pending: "待处理",
  running: "处理中",
  success: "已完成",
  error: "处理失败",
  timeout: "超时",
  interrupted: "已中断",
};

type Message = {
  id: number | string;
  role: "customer" | "assistant" | "system";
  text: string;
  ticketId?: string;
  failed?: boolean;
};

const initialMessages: Message[] = [
  { id: 1, role: "assistant", text: "你好，我是客服助手。你可以告诉我遇到的问题，我会尽力帮你。" },
];

export default function Home() {
  const [userId, setUserId] = useState("");
  const [loginValue, setLoginValue] = useState("");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTicketIds, setActiveTicketIds] = useState<Set<string>>(new Set());
  const [isEventConnected, setIsEventConnected] = useState(false);
  const [hasLoadedSession, setHasLoadedSession] = useState(false);
  const scrollArea = useRef<HTMLElement>(null);
  const seenAcceptedTickets = useRef(new Set<string>());
  const seenEventIds = useRef(new Set<string>());
  const ticketStatuses = useRef(new Map<string, string>());

  const showAcceptedTicket = useCallback((ticketId: string, status: string) => {
    if (seenAcceptedTickets.current.has(ticketId)) return;
    seenAcceptedTickets.current.add(ticketId);
    setMessages((current) => [...current, {
      id: `accepted:${ticketId}`,
      role: "system",
      ticketId,
      text: `工单 ${ticketId} 已受理 · ${statusLabels[status] ?? status}。`,
    }]);
  }, []);

  const hasSeenEvent = useCallback((eventId: string) => {
    if (!eventId || seenEventIds.current.has(eventId)) return true;
    seenEventIds.current.add(eventId);
    return false;
  }, []);

  const showEventError = useCallback((message: string) => {
    setMessages((current) => [...current, {
      id: `event-error:${Date.now()}`,
      role: "system",
      failed: true,
      text: message,
    }]);
  }, []);

  useEffect(() => {
    if (!userId) {
      setIsEventConnected(false);
      return;
    }

    const source = new EventSource(`/api/events?userId=${encodeURIComponent(userId)}`);
    source.onopen = () => setIsEventConnected(true);
    source.onerror = () => setIsEventConnected(false);

    source.addEventListener("message.accepted", (rawEvent) => {
      const event = rawEvent as MessageEvent<string>;
      if (hasSeenEvent(event.lastEventId)) return;
      try {
        const data = JSON.parse(event.data) as { ticket_id?: string; status?: string };
        if (!data.ticket_id) return;
        showAcceptedTicket(data.ticket_id, data.status ?? "pending");
        const status = ticketStatuses.current.get(data.ticket_id);
        if (!status || !terminalStatuses.has(status)) {
          setActiveTicketIds((current) => new Set(current).add(data.ticket_id!));
        }
      } catch {
        showEventError("收到无法识别的工单受理事件。");
      }
    });

    source.addEventListener("ticket.status", (rawEvent) => {
      const event = rawEvent as MessageEvent<string>;
      if (hasSeenEvent(event.lastEventId)) return;
      try {
        const data = JSON.parse(event.data) as { ticket_id?: string; status?: string };
        if (!data.ticket_id || !data.status) return;
        const ticketId = data.ticket_id;
        const status = data.status;
        ticketStatuses.current.set(ticketId, status);
        setActiveTicketIds((current) => {
          const next = new Set(current);
          if (terminalStatuses.has(status)) next.delete(ticketId);
          else next.add(ticketId);
          return next;
        });
        setMessages((current) => [...current, {
          id: `status:${event.lastEventId}`,
          role: "system",
          ticketId,
          failed: status === "error" || status === "timeout" || status === "interrupted",
          text: `工单 ${ticketId}：${statusLabels[status] ?? status}。`,
        }]);
      } catch {
        showEventError("收到无法识别的工单状态事件。");
      }
    });

    source.addEventListener("assistant.message", (rawEvent) => {
      const event = rawEvent as MessageEvent<string>;
      if (hasSeenEvent(event.lastEventId)) return;
      try {
        const data = JSON.parse(event.data) as { ticket_id?: string; message?: string };
        if (!data.ticket_id || !data.message?.trim()) return;
        setMessages((current) => [...current, {
          id: `assistant:${event.lastEventId}`,
          role: "assistant",
          ticketId: data.ticket_id,
          text: data.message!,
        }]);
      } catch {
        showEventError("收到无法识别的客服回复事件。");
      }
    });

    return () => source.close();
  }, [userId, hasSeenEvent, showAcceptedTicket, showEventError]);

  useEffect(() => {
    const savedUserId = window.sessionStorage.getItem("uda-user-id");
    const timer = window.setTimeout(() => {
      if (savedUserId) setUserId(savedUserId);
      setHasLoadedSession(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    scrollArea.current?.scrollTo({ top: scrollArea.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedId = loginValue.trim();
    if (!trimmedId) return;
    window.sessionStorage.setItem("uda-user-id", trimmedId);
    setUserId(trimmedId);
  }

  function signOut() {
    window.sessionStorage.removeItem("uda-user-id");
    setUserId("");
    setLoginValue("");
    setDraft("");
    setMessages(initialMessages);
    setActiveTicketIds(new Set());
    seenAcceptedTickets.current.clear();
    seenEventIds.current.clear();
    ticketStatuses.current.clear();
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || !isEventConnected || isSubmitting || activeTicketIds.size > 0) return;

    const messageId = Date.now();
    setMessages((current) => [...current, { id: messageId, role: "customer", text }]);
    setDraft("");
    setIsSubmitting(true);
    let acceptedTicketId: string | null = null;

    try {
      const response = await fetch("/api/tickets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, message: text }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof result.detail === "string" ? result.detail : "提交失败，请稍后重试。";
        setMessages((current) => [...current, {
          id: messageId + 1,
          role: "system",
          failed: true,
          text: `提交失败：${detail}`,
        }]);
        return;
      }
      if (typeof result.ticket_id !== "string" || typeof result.status !== "string") {
        throw new Error("后端响应格式不符合工单接口约定。");
      }
      const ticketId = result.ticket_id;
      acceptedTicketId = ticketId;
      showAcceptedTicket(ticketId, result.status);
      if (!terminalStatuses.has(ticketStatuses.current.get(ticketId) ?? "")) {
        setActiveTicketIds((current) => new Set(current).add(ticketId));
      }
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: messageId + (acceptedTicketId ? 2 : 1),
          role: "system",
          failed: true,
          text: acceptedTicketId
            ? `工单 ${acceptedTicketId} 已受理，但没有取回处理结果：${error instanceof Error ? error.message : "请检查后端服务连接。"}`
            : `提交状态未确认：${error instanceof Error ? error.message : "请检查后端服务连接。"}`,
        },
      ]);
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  if (!hasLoadedSession) return <main className={styles.loading} aria-label="正在加载" />;

  if (!userId) {
    return (
      <main className={styles.loginPage}>
        <section className={styles.loginCard} aria-labelledby="login-title">
          <div className={styles.brandMark} aria-hidden="true">U</div>
          <p className={styles.eyebrow}>CUSTOMER SUPPORT</p>
          <h1 id="login-title">欢迎回来</h1>
          <p className={styles.loginDescription}>输入用户 ID，开始与客服助手对话。</p>
          <form className={styles.loginForm} onSubmit={signIn}>
            <label htmlFor="user-id">用户 ID</label>
            <input autoComplete="username" autoFocus id="user-id" onChange={(event) => setLoginValue(event.target.value)} placeholder="例如：user-001" value={loginValue} />
            <button className={styles.loginButton} disabled={!loginValue.trim()} type="submit">
              进入对话 <span aria-hidden="true">→</span>
            </button>
          </form>
          <p className={styles.demoNote}>演示界面 · 尚未连接身份验证服务</p>
        </section>
      </main>
    );
  }

  return (
    <main className={styles.page}>
      <section className={styles.chatPanel} aria-label="客服对话">
        <header className={styles.header}>
          <div className={styles.headerTitle}>
            <span className={styles.headerIcon} aria-hidden="true">U</span>
            <div><h1>客服助手</h1><p><span className={styles.onlineDot} />{isEventConnected ? "实时消息已连接" : "正在连接实时消息…"}</p></div>
          </div>
          <div className={styles.headerActions}>
            <span className={styles.userBadge} title={`当前用户：${userId}`}>
              <span className={styles.userAvatar} aria-hidden="true">{userId.slice(0, 1).toUpperCase()}</span>
              <span className={styles.userLabel}>{userId}</span>
            </span>
            <button className={styles.signOut} disabled={isSubmitting || activeTicketIds.size > 0} onClick={signOut} type="button">切换用户</button>
          </div>
        </header>

        <section className={styles.conversation} ref={scrollArea} aria-label="消息记录" aria-live="polite">
          <p className={styles.dateMarker}>今天</p>
          {messages.map((message) => (
            <div className={`${styles.messageRow} ${message.role === "customer" ? styles.customerRow : message.role === "system" ? styles.systemRow : ""}`} key={message.id}>
              {message.role === "assistant" && <span className={styles.assistantAvatar} aria-hidden="true">U</span>}
              <article className={`${styles.message} ${message.role === "customer" ? styles.customerMessage : message.role === "system" ? `${styles.systemMessage} ${message.failed ? styles.failedMessage : ""}` : styles.assistantMessage}`}>
                {message.text}
              </article>
            </div>
          ))}
        </section>

        <footer className={styles.composerArea}>
          <form className={styles.composer} onSubmit={sendMessage}>
            <label className={styles.srOnly} htmlFor="message">输入消息</label>
            <textarea disabled={!isEventConnected || isSubmitting || activeTicketIds.size > 0} id="message" onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder={!isEventConnected ? "正在连接实时消息…" : isSubmitting ? "正在提交到客服工作流…" : activeTicketIds.size > 0 ? "客服工作流处理中…" : "输入消息…"} rows={1} value={draft} />
            <button className={styles.sendButton} disabled={!isEventConnected || !draft.trim() || isSubmitting || activeTicketIds.size > 0} aria-label="发送消息" type="submit">
              <svg aria-hidden="true" viewBox="0 0 24 24" fill="none"><path d="M12 19V5m0 0L6 11m6-6 6 6" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" /></svg>
            </button>
          </form>
          <p className={styles.composerHint}>Enter 发送 · Shift + Enter 换行</p>
          <p className={styles.demoDisclaimer}>{isSubmitting || activeTicketIds.size > 0 ? "客服工作流处理中，回复将通过实时消息到达" : "消息将提交到客服工作流并通过实时消息返回"}</p>
        </footer>
      </section>
    </main>
  );
}
