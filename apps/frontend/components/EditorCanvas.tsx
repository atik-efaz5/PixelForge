"use client";

import type { EditorTool, ImageDimensions } from "@/types/api";
import { computeDisplayLayout, pointerToImageCoords } from "@/lib/coordinates";
import { drawMaskOverlay } from "@/lib/mask";
import { useCallback, useEffect, useRef } from "react";

interface EditorCanvasProps {
  imageUrl: string | null;
  imageSize: ImageDimensions | null;
  mask: Uint8Array | null;
  tool: EditorTool;
  brushRadius: number;
  eraserRadius: number;
  featherRadius: number;
  showOverlay: boolean;
  showMaskOnly: boolean;
  disabled: boolean;
  onPointSelect: (x: number, y: number) => void;
  onBrushStroke: (x: number, y: number, mode: "brush" | "erase") => void;
  onStrokeStart: () => void;
  onStrokeEnd: () => void;
}

export function EditorCanvas({
  imageUrl,
  imageSize,
  mask,
  tool,
  brushRadius,
  eraserRadius,
  featherRadius,
  showOverlay,
  showMaskOnly,
  disabled,
  onPointSelect,
  onBrushStroke,
  onStrokeStart,
  onStrokeEnd,
}: EditorCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const layoutRef = useRef<ReturnType<typeof computeDisplayLayout> | null>(null);
  const paintingRef = useRef(false);
  const strokeStartedRef = useRef(false);

  const activeRadius = tool === "erase" ? eraserRadius : brushRadius;

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imageRef.current;
    if (!canvas || !container || !imageSize) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const layout = computeDisplayLayout(
      { width: rect.width, height: rect.height },
      imageSize
    );
    layoutRef.current = layout;

    if (showMaskOnly && mask) {
      ctx.fillStyle = "#0f1115";
      ctx.fillRect(0, 0, rect.width, rect.height);
      const preview = document.createElement("canvas");
      preview.width = imageSize.width;
      preview.height = imageSize.height;
      const pctx = preview.getContext("2d");
      if (pctx) {
        const imageData = pctx.createImageData(imageSize.width, imageSize.height);
        for (let i = 0; i < mask.length; i += 1) {
          const v = mask[i] ? 255 : 0;
          const idx = i * 4;
          imageData.data[idx] = v;
          imageData.data[idx + 1] = v;
          imageData.data[idx + 2] = v;
          imageData.data[idx + 3] = 255;
        }
        pctx.putImageData(imageData, 0, 0);
        ctx.drawImage(
          preview,
          layout.offsetX,
          layout.offsetY,
          layout.renderedWidth,
          layout.renderedHeight
        );
      }
    } else if (img) {
      ctx.drawImage(
        img,
        layout.offsetX,
        layout.offsetY,
        layout.renderedWidth,
        layout.renderedHeight
      );
      if (mask && showOverlay) {
        drawMaskOverlay(ctx, mask, imageSize.width, imageSize.height, layout, {
          featherRadius: featherRadius > 0 ? featherRadius : undefined,
        });
      }
    }
  }, [featherRadius, imageSize, mask, showMaskOnly, showOverlay]);

  useEffect(() => {
    if (!imageUrl) {
      imageRef.current = null;
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      redraw();
    };
    img.src = imageUrl;
    return () => {
      img.onload = null;
    };
  }, [imageUrl, redraw]);

  useEffect(() => {
    redraw();
  }, [mask, redraw]);

  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [redraw]);

  const handlePointer = useCallback(
    (clientX: number, clientY: number, isStroke: boolean) => {
      const canvas = canvasRef.current;
      const layout = layoutRef.current;
      if (!canvas || !layout || disabled || !imageSize) return;

      const rect = canvas.getBoundingClientRect();
      const coords = pointerToImageCoords(
        clientX - rect.left,
        clientY - rect.top,
        layout,
        imageSize
      );
      if (!coords) return;

      if (tool === "select" && !isStroke) {
        onPointSelect(coords.x, coords.y);
      } else if (tool === "brush" || tool === "erase") {
        onBrushStroke(coords.x, coords.y, tool === "brush" ? "brush" : "erase");
      }
    },
    [disabled, imageSize, onBrushStroke, onPointSelect, tool]
  );

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!imageUrl || disabled) return;
    paintingRef.current = true;
    strokeStartedRef.current = false;
    event.currentTarget.setPointerCapture(event.pointerId);
    if (tool === "brush" || tool === "erase") {
      onStrokeStart();
      strokeStartedRef.current = true;
    }
    handlePointer(event.clientX, event.clientY, tool !== "select");
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!paintingRef.current || tool === "select") return;
    handlePointer(event.clientX, event.clientY, true);
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (paintingRef.current && strokeStartedRef.current) {
      onStrokeEnd();
    }
    paintingRef.current = false;
    strokeStartedRef.current = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const cursor =
    tool === "select"
      ? "crosshair"
      : tool === "brush"
        ? "cell"
        : "not-allowed";

  return (
    <div
      ref={containerRef}
      className="editor-canvas-wrap"
      style={{
        flex: 1,
        minHeight: 420,
        background: "#1a1d24",
        borderRadius: 8,
        border: "1px solid #2a2f3a",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {!imageUrl ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            color: "#9aa3b2",
            fontSize: 15,
          }}
        >
          Upload an image to begin
        </div>
      ) : (
        <canvas
          ref={canvasRef}
          aria-label="Image editor canvas"
          role="img"
          style={{
            display: "block",
            width: "100%",
            height: "100%",
            cursor,
            touchAction: "none",
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
        />
      )}
      {imageUrl && tool !== "select" ? (
        <div
          style={{
            position: "absolute",
            left: 12,
            bottom: 12,
            padding: "4px 8px",
            borderRadius: 4,
            background: "rgba(0,0,0,0.55)",
            fontSize: 12,
            color: "#cbd5e1",
          }}
        >
          {tool === "erase" ? "Eraser" : "Brush"}: {activeRadius}px
        </div>
      ) : null}
    </div>
  );
}
