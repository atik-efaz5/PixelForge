import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PixelForge",
  description: "MVP image editor — segment and inpaint objects locally.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
