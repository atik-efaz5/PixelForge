"use client";

import type { EditorTool, ImageDimensions } from "@/types/api";
import { pointerToImageCoords } from "@/lib/coordinates";
import {
  computeViewLayout,
  fitViewport,
  type CanvasViewport,
  zoomIn,
  zoomOut,
  zoomViewportAtPoint,
} from "@/lib/canvasView";
import { drawMaskOverlay } from "@/lib/mask";
import { useCallback, useEffect, useRef, useState } from "react";

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
  viewport: CanvasViewport;
  onViewportChange: (viewport: CanvasViewport) => void;
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
  viewport,
  onViewportChange,
  onPointSelect,
  onBrushStroke,
  onStrokeStart,
  onStrokeEnd,
}: EditorCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const layoutRef = useRef<ReturnType<typeof computeViewLayout> | null>(null);
  const paintingRef = useRef(false);
  const panningRef = useRef(false);
  const strokeStartedRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const spaceDownRef = useRef(false);
  const [pointerPos, setPointerPos] = useState<{
    screenX: number;
    screenY: number;
    imageX: number;
    imageY: number;
  } | null>(null);
  const [isPanning, setIsPanning] = useState(false);

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

    const layout = computeViewLayout(
      { width: rect.width, height: rect.height },
      imageSize,
      viewport
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

    if (
      pointerPos &&
      (tool === "brush" || tool === "erase") &&
      !panningRef.current
    ) {
      const radiusScreen = activeRadius * layout.scale;
      ctx.beginPath();
      ctx.arc(pointerPos.screenX, pointerPos.screenY, radiusScreen, 0, Math.PI * 2);
      ctx.strokeStyle =
        tool === "erase" ? "rgba(248, 113, 113, 0.9)" : "rgba(56, 189, 248, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pointerPos.screenX, pointerPos.screenY, 2, 0, Math.PI * 2);
      ctx.fillStyle = tool === "erase" ? "#f87171" : "#38bdf8";
      ctx.fill();
    }
  }, [
    activeRadius,
    featherRadius,
    imageSize,
    mask,
    pointerPos,
    showMaskOnly,
    showOverlay,
    tool,
    viewport,
  ]);

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
  }, [mask, redraw, viewport]);

  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [redraw]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        spaceDownRef.current = true;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        spaceDownRef.current = false;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !imageSize) return;

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = container.getBoundingClientRect();
      const layout = layoutRef.current;
      if (!layout) return;
      const factor = event.deltaY > 0 ? 1 / 1.12 : 1.12;
      onViewportChange(
        zoomViewportAtPoint(
          viewport,
          factor,
          event.clientX - rect.left,
          event.clientY - rect.top,
          layout,
          { width: rect.width, height: rect.height },
          imageSize
        )
      );
    };

    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, [imageSize, onViewportChange, viewport]);

  const handlePointer = useCallback(
    (clientX: number, clientY: number, isStroke: boolean) => {
      const canvas = canvasRef.current;
      const layout = layoutRef.current;
      if (!canvas || !layout || disabled || !imageSize) return;

      const rect = canvas.getBoundingClientRect();
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      const coords = pointerToImageCoords(localX, localY, layout, imageSize);
      if (coords) {
        setPointerPos({
          screenX: localX,
          screenY: localY,
          imageX: coords.x,
          imageY: coords.y,
        });
      } else {
        setPointerPos(null);
      }
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
    const isMiddle = event.button === 1;
    const isPan = isMiddle || spaceDownRef.current;
    if (isPan) {
      panningRef.current = true;
      setIsPanning(true);
      panStartRef.current = {
        x: event.clientX,
        y: event.clientY,
        panX: viewport.panX,
        panY: viewport.panY,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }

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
    if (panningRef.current) {
      const dx = event.clientX - panStartRef.current.x;
      const dy = event.clientY - panStartRef.current.y;
      onViewportChange({
        ...viewport,
        panX: panStartRef.current.panX + dx,
        panY: panStartRef.current.panY + dy,
      });
      return;
    }
    handlePointer(event.clientX, event.clientY, paintingRef.current && tool !== "select");
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (panningRef.current) {
      panningRef.current = false;
      setIsPanning(false);
      event.currentTarget.releasePointerCapture(event.pointerId);
      return;
    }
    if (paintingRef.current && strokeStartedRef.current) {
      onStrokeEnd();
    }
    paintingRef.current = false;
    strokeStartedRef.current = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const onPointerLeave = () => {
    setPointerPos(null);
    setIsPanning(false);
    panningRef.current = false;
    if (paintingRef.current && strokeStartedRef.current) {
      onStrokeEnd();
    }
    paintingRef.current = false;
    panningRef.current = false;
    strokeStartedRef.current = false;
  };

  const cursor = isPanning
    ? "grabbing"
    : spaceDownRef.current
      ? "grab"
      : tool === "select"
        ? "crosshair"
        : tool === "brush"
          ? "none"
          : "none";

  return (
    <div
      ref={containerRef}
      className="editor-canvas-wrap"
      style={{
        flex: 1,
        minHeight: 360,
        background: "#1a1d24",
        borderRadius: 8,
        border: "1px solid #2a2f3a",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {imageUrl ? (
        <div
          className="editor-canvas-toolbar"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "8px 10px",
            borderBottom: "1px solid #2a2f3a",
            background: "#12151b",
            flexWrap: "wrap",
          }}
        >
          <ToolbarButton
            label="Zoom out"
            disabled={disabled}
            onClick={() => onViewportChange(zoomOut(viewport))}
          >
            −
          </ToolbarButton>
          <span style={{ fontSize: 12, color: "#9aa3b2", minWidth: 48, textAlign: "center" }}>
            {Math.round(viewport.zoom * 100)}%
          </span>
          <ToolbarButton
            label="Zoom in"
            disabled={disabled}
            onClick={() => onViewportChange(zoomIn(viewport))}
          >
            +
          </ToolbarButton>
          <ToolbarButton
            label="Fit to screen"
            disabled={disabled}
            onClick={() => onViewportChange(fitViewport())}
          >
            Fit
          </ToolbarButton>
          <ToolbarButton
            label="Reset view"
            disabled={disabled}
            onClick={() => onViewportChange(fitViewport())}
          >
            Reset
          </ToolbarButton>
          <span style={{ fontSize: 11, color: "#5c6573", marginLeft: "auto" }}>
            Scroll to zoom · Space+drag to pan
          </span>
        </div>
      ) : null}

      <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
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
            onPointerLeave={onPointerLeave}
          />
        )}

        {imageUrl && pointerPos ? (
          <div
            style={{
              position: "absolute",
              right: 12,
              bottom: 12,
              padding: "4px 8px",
              borderRadius: 4,
              background: "rgba(0,0,0,0.6)",
              fontSize: 11,
              color: "#cbd5e1",
              fontVariantNumeric: "tabular-nums",
              pointerEvents: "none",
            }}
          >
            {pointerPos.imageX}, {pointerPos.imageY}
            {tool !== "select" ? ` · ${activeRadius}px` : ""}
          </div>
        ) : null}

        {imageUrl && tool !== "select" && !pointerPos ? (
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
              pointerEvents: "none",
            }}
          >
            {tool === "erase" ? "Eraser" : "Brush"}: {activeRadius}px
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ToolbarButton({
  children,
  label,
  disabled,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      style={{
        padding: "4px 10px",
        borderRadius: 4,
        border: "1px solid #3a4150",
        background: "#1f2430",
        color: "#e8eaed",
        fontSize: 13,
        minWidth: 32,
      }}
    >
      {children}
    </button>
  );
}
