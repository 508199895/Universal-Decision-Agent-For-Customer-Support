import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "客服助手 | Universal Decision Agent",
  description: "与客服助手开始一段对话。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
