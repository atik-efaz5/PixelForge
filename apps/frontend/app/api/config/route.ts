import { NextResponse } from "next/server";

/** Server-side API base URL for Vercel (no client rebuild when tunnel URL changes). */
export function GET() {
  const apiBaseUrl =
    process.env.PIXELFORGE_API_BASE_URL?.trim() ||
    process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ||
    "";

  return NextResponse.json(
    { apiBaseUrl },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    }
  );
}
