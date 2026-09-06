"use client";

import type { EditorTool, ImageDimensions } from "@/types/api";
import {
  resetInteractionState,
  shouldPaintStroke,
  shouldTriggerPointSelect,
} from "@/lib/canvasPointer";
import { pointerToImageCoords } from "@/lib/coordinates";
import {
  computeViewLayout,
  fitViewport,
  type CanvasViewport,
  zoomIn,
  zoomOut,
  zoomViewportAtPoint,
} from "@/lib/canvasView";
import { shouldReloadCanvasImage } from "@/lib/canvasImageSource";
import {
  canvasBitmapSize,
  readStageSize,
  shouldSkipStageRedraw,
  type StageSize,
} from "@/lib/canvasStage";
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
  onImageDecodeError?: () => void;
}

type PointerHud = {
  screenX: number;
  screenY: number;
  imageX: number;
  imageY: number;
};

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
  onImageDecodeError,
}: EditorCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastStageSizeRef = useRef<StageSize | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const loadedSourceUrlRef = useRef<string | null>(null);
  const imageUrlRef = useRef<string | null>(imageUrl);
  imageUrlRef.current = imageUrl;
  const layoutRef = useRef<ReturnType<typeof computeViewLayout> | null>(null);
  const paintingRef = useRef(false);
  const panningRef = useRef(false);
  const strokeStartedRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const spaceDownRef = useRef(false);
  const activePointerIdRef = useRef<number | null>(null);
  const pointerPosRef = useRef<PointerHud | null>(null);
  const viewportRef = useRef(viewport);
  const redrawRef = useRef<() => void>(() => {});
  const onImageDecodeErrorRef = useRef(onImageDecodeError);
  onImageDecodeErrorRef.current = onImageDecodeError;
  const [pointerHud, setPointerHud] = useState<PointerHud | null>(null);
  const [isPanning, setIsPanning] = useState(false);

  viewportRef.current = viewport;

  const activeRadius = tool === "erase" ? eraserRadius : brushRadius;

  const releasePointerCapture = useCallback(() => {
    const canvas = canvasRef.current;
    const pointerId = activePointerIdRef.current;
    if (canvas && pointerId !== null && canvas.hasPointerCapture(pointerId)) {
      canvas.releasePointerCapture(pointerId);
    }
    activePointerIdRef.current = null;
  }, []);

  const endInteraction = useCallback(
    (options?: { endStroke?: boolean }) => {
      if (options?.endStroke && paintingRef.current && strokeStartedRef.current) {
        onStrokeEnd();
      }
      const idle = resetInteractionState();
      panningRef.current = idle.isPanning;
      paintingRef.current = idle.isPainting;
      strokeStartedRef.current = idle.strokeStarted;
      setIsPanning(false);
      releasePointerCapture();
    },
    [onStrokeEnd, releasePointerCapture]
  );

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    const img = imageRef.current;
    if (!canvas || !stage || !imageSize) return;

    const stageSize = readStageSize(stage);
    if (stageSize.width <= 0 || stageSize.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = stageSize.width;
    const cssHeight = stageSize.height;
    const bitmap = canvasBitmapSize(cssWidth, cssHeight, dpr);

    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    lastStageSizeRef.current = stageSize;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const layout = computeViewLayout(
      { width: cssWidth, height: cssHeight },
      imageSize,
      viewportRef.current
    );
    layoutRef.current = layout;

    if (showMaskOnly && mask) {
      ctx.fillStyle = "#0f1115";
      ctx.fillRect(0, 0, cssWidth, cssHeight);
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

    const pointerPos = pointerPosRef.current;
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
    showMaskOnly,
    showOverlay,
    tool,
  ]);

  redrawRef.current = redraw;

  useEffect(() => {
    const sourceUrl = imageUrl;
    if (!sourceUrl) {
      imageRef.current = null;
      loadedSourceUrlRef.current = null;
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    if (!shouldReloadCanvasImage(loadedSourceUrlRef.current, sourceUrl, !!imageRef.current)) {
      redrawRef.current();
      return;
    }

    const img = new Image();
    img.onload = () => {
      if (imageUrlRef.current !== sourceUrl) return;
      imageRef.current = img;
      loadedSourceUrlRef.current = sourceUrl;
      requestAnimationFrame(() => {
        redrawRef.current();
      });
    };
    img.onerror = () => {
      if (imageUrlRef.current !== sourceUrl) return;
      if (loadedSourceUrlRef.current === sourceUrl && imageRef.current) {
        redrawRef.current();
        return;
      }
      onImageDecodeErrorRef.current?.();
    };
    img.src = sourceUrl;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl]);

  useEffect(() => {
    redraw();
  }, [mask, redraw, viewport]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      const box = entry?.contentBoxSize?.[0];
      const next: StageSize = box
        ? { width: box.inlineSize, height: box.blockSize }
        : readStageSize(stage);
      if (shouldSkipStageRedraw(lastStageSizeRef.current, next)) return;
      redrawRef.current();
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [redraw]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space" && !event.repeat) {
        spaceDownRef.current = true;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        spaceDownRef.current = false;
        if (panningRef.current) {
          endInteraction();
        }
      }
    };
    const onBlur = () => {
      spaceDownRef.current = false;
      endInteraction({ endStroke: true });
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [endInteraction]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !imageSize) return;

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const layout = layoutRef.current;
      if (!layout) return;
      const rect = stage.getBoundingClientRect();
      const size = readStageSize(stage);
      const factor = event.deltaY > 0 ? 1 / 1.12 : 1.12;
      onViewportChange(
        zoomViewportAtPoint(
          viewportRef.current,
          factor,
          event.clientX - rect.left,
          event.clientY - rect.top,
          layout,
          { width: size.width, height: size.height },
          imageSize
        )
      );
    };

    stage.addEventListener("wheel", onWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onWheel);
  }, [imageSize, onViewportChange]);

  const updatePointerHud = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      const layout = layoutRef.current;
      if (!canvas || !layout || !imageSize) {
        pointerPosRef.current = null;
        setPointerHud(null);
        return null;
      }

      const rect = canvas.getBoundingClientRect();
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      const coords = pointerToImageCoords(localX, localY, layout, imageSize);
      if (!coords) {
        pointerPosRef.current = null;
        setPointerHud(null);
        return null;
      }

      const hud: PointerHud = {
        screenX: localX,
        screenY: localY,
        imageX: coords.x,
        imageY: coords.y,
      };
      pointerPosRef.current = hud;
      setPointerHud(hud);
      return coords;
    },
    [imageSize]
  );

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!imageUrl || disabled) return;
    if (event.button !== 0 && event.button !== 1) return;

    const isMiddle = event.button === 1;
    const isPan = isMiddle || spaceDownRef.current;
    if (isPan) {
      event.preventDefault();
      panningRef.current = true;
      setIsPanning(true);
      panStartRef.current = {
        x: event.clientX,
        y: event.clientY,
        panX: viewportRef.current.panX,
        panY: viewportRef.current.panY,
      };
      activePointerIdRef.current = event.pointerId;
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }

    activePointerIdRef.current = event.pointerId;
    event.currentTarget.setPointerCapture(event.pointerId);

    const coords = updatePointerHud(event.clientX, event.clientY);
    if (!coords) return;

    if (shouldTriggerPointSelect(tool, "down")) {
      onPointSelect(coords.x, coords.y);
      return;
    }

    if (tool === "brush" || tool === "erase") {
      paintingRef.current = true;
      strokeStartedRef.current = false;
      onStrokeStart();
      strokeStartedRef.current = true;
      onBrushStroke(coords.x, coords.y, tool === "brush" ? "brush" : "erase");
    }
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (panningRef.current) {
      const dx = event.clientX - panStartRef.current.x;
      const dy = event.clientY - panStartRef.current.y;
      onViewportChange({
        ...viewportRef.current,
        panX: panStartRef.current.panX + dx,
        panY: panStartRef.current.panY + dy,
      });
      return;
    }

    const coords = updatePointerHud(event.clientX, event.clientY);
    if (tool === "brush" || tool === "erase") {
      redraw();
    }

    if (!coords || !shouldPaintStroke(tool, paintingRef.current)) {
      return;
    }
    onBrushStroke(coords.x, coords.y, tool === "brush" ? "brush" : "erase");
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (activePointerIdRef.current !== null && event.pointerId !== activePointerIdRef.current) {
      return;
    }
    endInteraction({ endStroke: true });
  };

  const onPointerCancel = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (activePointerIdRef.current !== null && event.pointerId !== activePointerIdRef.current) {
      return;
    }
    endInteraction({ endStroke: true });
  };

  const onPointerLeave = () => {
    pointerPosRef.current = null;
    setPointerHud(null);
    redraw();
    endInteraction({ endStroke: true });
  };

  const cursor = isPanning
    ? "grabbing"
    : spaceDownRef.current
      ? "grab"
      : tool === "select"
        ? "crosshair"
        : "none";

  return (
    <div
      ref={containerRef}
      className="editor-canvas-wrap"
      style={{
        flex: 1,
        minHeight: 0,
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

      <div
        ref={stageRef}
        className="editor-canvas-stage"
        style={{ flex: 1, position: "relative", minHeight: 0 }}
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
            onPointerCancel={onPointerCancel}
            onPointerLeave={onPointerLeave}
            onContextMenu={(event) => event.preventDefault()}
          />
        )}

        {imageUrl && pointerHud ? (
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
            {pointerHud.imageX}, {pointerHud.imageY}
            {tool !== "select" ? ` · ${activeRadius}px` : ""}
          </div>
        ) : null}

        {imageUrl && tool !== "select" && !pointerHud ? (
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
