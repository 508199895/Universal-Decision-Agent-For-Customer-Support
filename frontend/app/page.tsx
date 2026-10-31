"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

type Message = { id: number; role: "customer" | "assistant" | "system"; text: string; failed?: boolean };

const initialMessages: Message[] = [
  { id: 1, role: "assistant", text: "你好，我是客服助手。你可以告诉我遇到的问题，我会尽力帮你。" },
];

export default function Home() {
  const [userId, setUserId] = useState("");
  const [loginValue, setLoginValue] = useState("");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [isSending, setIsSending] = useState(false);
  const [hasLoadedSession, setHasLoadedSession] = useState(false);
  const scrollArea = useRef<HTMLElement>(null);

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
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || isSending) return;

    const messageId = Date.now();
    setMessages((current) => [...current, { id: messageId, role: "customer", text }]);
    setDraft("");
    setIsSending(true);
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
      const statusLabels: Record<string, string> = {
        pending: "待处理",
        running: "处理中",
        success: "已完成",
        error: "处理失败",
        timeout: "超时",
        interrupted: "已中断",
      };

      setMessages((current) => [
        ...current,
        {
          id: messageId + 1,
          role: "system",
          text: `已接入客服工作流 · 工单 ${ticketId} · ${statusLabels[result.status] ?? result.status}。正在等待回复。`,
        },
      ]);

      while (true) {
        const statusResponse = await fetch(
          `/api/tickets/${encodeURIComponent(ticketId)}?userId=${encodeURIComponent(userId)}`,
          { cache: "no-store" },
        );
        const ticket = await statusResponse.json().catch(() => ({}));
        if (!statusResponse.ok) {
          const detail = typeof ticket.detail === "string" ? ticket.detail : "查询处理结果失败。";
          throw new Error(detail);
        }

        if (["success", "error", "timeout", "interrupted"].includes(ticket.status)) {
          if (ticket.status === "success" && typeof ticket.last_ai_message === "string" && ticket.last_ai_message.trim()) {
            setMessages((current) => [...current, {
              id: messageId + 2,
              role: "assistant",
              text: ticket.last_ai_message,
            }]);
          } else {
            const outcome = ticket.status === "success"
              ? "工作流已完成，但没有生成可展示的文本回复。"
              : ticket.status === "error"
                ? `工作流处理失败：${ticket.error || "未知错误。"}`
                : ticket.status === "timeout"
                  ? "工作流处理超时，请稍后重试。"
                  : "工作流已中断。";
            setMessages((current) => [...current, {
              id: messageId + 2,
              role: "system",
              failed: ticket.status !== "success",
              text: outcome,
            }]);
          }
          return;
        }

        await new Promise((resolve) => window.setTimeout(resolve, 1500));
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
      setIsSending(false);
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
            <div><h1>客服助手</h1><p><span className={styles.onlineDot} />在线为你服务</p></div>
          </div>
          <div className={styles.headerActions}>
            <span className={styles.userBadge} title={`当前用户：${userId}`}>
              <span className={styles.userAvatar} aria-hidden="true">{userId.slice(0, 1).toUpperCase()}</span>
              <span className={styles.userLabel}>{userId}</span>
            </span>
            <button className={styles.signOut} disabled={isSending} onClick={signOut} type="button">切换用户</button>
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
            <textarea disabled={isSending} id="message" onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder={isSending ? "正在提交到客服工作流…" : "输入消息…"} rows={1} value={draft} />
            <button className={styles.sendButton} disabled={!draft.trim() || isSending} aria-label="发送消息" type="submit">
              <svg aria-hidden="true" viewBox="0 0 24 24" fill="none"><path d="M12 19V5m0 0L6 11m6-6 6 6" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" /></svg>
            </button>
          </form>
          <p className={styles.composerHint}>Enter 发送 · Shift + Enter 换行</p>
          <p className={styles.demoDisclaimer}>{isSending ? "客服工作流处理中，完成后会在此显示回复" : "消息将提交到客服工作流并等待处理结果"}</p>
        </footer>
      </section>
    </main>
  );
}
