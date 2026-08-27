"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { inpaint, segment, selectByText } from "@/lib/api";
import {
  createEmptyMask,
  decodeMaskPng,
  encodeMaskPng,
  maskHasInpaint,
  maskPreviewUrl,
  paintBrush,
} from "@/lib/mask";
import type {
  DetectionInfo,
  EditorStatus,
  EditorTool,
  ImageDimensions,
  InpaintBackend,
} from "@/types/api";
import { ControlPanel } from "@/components/ControlPanel";
import { EditorCanvas } from "@/components/EditorCanvas";
import { ResultPanel } from "@/components/ResultPanel";

function revokeIfObjectUrl(url: string | null) {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

export function ImageEditor() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<ImageDimensions | null>(null);
  const [mask, setMask] = useState<Uint8Array | null>(null);
  const [maskPreview, setMaskPreview] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [tool, setTool] = useState<EditorTool>("select");
  const [brushRadius, setBrushRadius] = useState(24);
  const [backend] = useState<InpaintBackend>("moebius");
  const [textPrompt, setTextPrompt] = useState("");
  const [detections, setDetections] = useState<DetectionInfo[]>([]);
  const [detectionIndex, setDetectionIndex] = useState(0);
  const [status, setStatus] = useState<EditorStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const maskRef = useRef<Uint8Array | null>(null);

  const busy = status !== "idle";

  const updateMaskPreview = useCallback(
    (nextMask: Uint8Array | null, size: ImageDimensions | null) => {
      if (!nextMask || !size) {
        setMaskPreview(null);
        return;
      }
      setMaskPreview(maskPreviewUrl(nextMask, size.width, size.height));
    },
    []
  );

  const resetSession = useCallback(
    (file: File, url: string, size: ImageDimensions) => {
      revokeIfObjectUrl(imageUrl);
      revokeIfObjectUrl(resultUrl);
      setImageFile(file);
      setImageUrl(url);
      setImageSize(size);
      setMask(null);
      maskRef.current = null;
      setMaskPreview(null);
      setResultUrl(null);
      setLatencyMs(null);
      setError(null);
      setTool("select");
      setTextPrompt("");
      setDetections([]);
      setDetectionIndex(0);
    },
    [imageUrl, resultUrl]
  );

  const handleUpload = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        resetSession(file, url, { width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        setError("Could not load the selected image.");
      };
      img.src = url;
    },
    [resetSession]
  );

  const applyMaskFromBlob = useCallback(
    async (blob: Blob) => {
      if (!imageSize) return;
      const decoded = await decodeMaskPng(blob, imageSize);
      maskRef.current = decoded;
      setMask(decoded);
      updateMaskPreview(decoded, imageSize);
      setTool("brush");
    },
    [imageSize, updateMaskPreview]
  );

  const handleFindObject = useCallback(async () => {
    if (!imageFile || !imageSize || busy) return;
    const prompt = textPrompt.trim();
    if (!prompt) {
      setError("Enter an object description to search.");
      return;
    }
    setStatus("grounding");
    setError(null);
    try {
      const { blob, metadata } = await selectByText(
        imageFile,
        prompt,
        detectionIndex
      );
      setDetections(metadata.detections || []);
      setDetectionIndex(metadata.detection_index ?? 0);
      await applyMaskFromBlob(blob);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Text selection failed.";
      setError(message);
    } finally {
      setStatus("idle");
    }
  }, [
    applyMaskFromBlob,
    busy,
    detectionIndex,
    imageFile,
    imageSize,
    textPrompt,
  ]);

  const handleDetectionIndexChange = useCallback(
    async (index: number) => {
      setDetectionIndex(index);
      if (!imageFile || !textPrompt.trim() || busy) return;
      setStatus("grounding");
      setError(null);
      try {
        const { blob, metadata } = await selectByText(
          imageFile,
          textPrompt.trim(),
          index
        );
        setDetections(metadata.detections || []);
        await applyMaskFromBlob(blob);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Could not switch detection.";
        setError(message);
      } finally {
        setStatus("idle");
      }
    },
    [applyMaskFromBlob, busy, imageFile, textPrompt]
  );

  const handlePointSelect = useCallback(
    async (x: number, y: number) => {
      if (!imageFile || !imageSize || busy) return;
      setStatus("segmenting");
      setError(null);
      try {
        const { blob } = await segment(imageFile, x, y);
        await applyMaskFromBlob(blob);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Segmentation failed.";
        setError(message);
      } finally {
        setStatus("idle");
      }
    },
    [applyMaskFromBlob, busy, imageFile, imageSize]
  );

  const handleBrushStroke = useCallback(
    (x: number, y: number, mode: "brush" | "erase") => {
      if (!imageSize || !maskRef.current) return;
      const next = new Uint8Array(maskRef.current);
      paintBrush(next, imageSize.width, imageSize.height, x, y, brushRadius, mode);
      maskRef.current = next;
      setMask(next);
      updateMaskPreview(next, imageSize);
    },
    [brushRadius, imageSize, updateMaskPreview]
  );

  const handleClearMask = useCallback(() => {
    if (!imageSize) return;
    const cleared = createEmptyMask(imageSize.width, imageSize.height);
    maskRef.current = cleared;
    setMask(cleared);
    updateMaskPreview(cleared, imageSize);
  }, [imageSize, updateMaskPreview]);

  const handleGenerate = useCallback(async () => {
    if (!imageFile || !imageSize || !maskRef.current || busy) return;
    if (!maskHasInpaint(maskRef.current)) {
      setError("Draw or select a mask region before generating.");
      return;
    }
    setStatus("generating");
    setError(null);
    try {
      const maskBlob = await encodeMaskPng(
        maskRef.current,
        imageSize.width,
        imageSize.height
      );
      const { blob, metadata } = await inpaint(imageFile, maskBlob, backend);
      revokeIfObjectUrl(resultUrl);
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
      setLatencyMs(metadata.latency_ms ?? null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Inpainting failed.";
      setError(message);
    } finally {
      setStatus("idle");
    }
  }, [backend, busy, imageFile, imageSize, resultUrl]);

  useEffect(() => {
    return () => {
      revokeIfObjectUrl(imageUrl);
      revokeIfObjectUrl(resultUrl);
    };
  }, [imageUrl, resultUrl]);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          padding: "14px 20px",
          borderBottom: "1px solid #2a2f3a",
          background: "#12151b",
        }}
      >
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em" }}>
          PixelForge
        </h1>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#7b8494" }}>
          Click to segment · refine mask · generate with Moebius
        </p>
      </header>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        <main
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
            padding: 20,
            gap: 0,
          }}
        >
          {error ? (
            <div
              role="alert"
              style={{
                marginBottom: 12,
                padding: "10px 12px",
                borderRadius: 6,
                background: "#3f1d1d",
                border: "1px solid #7f1d1d",
                color: "#fecaca",
                fontSize: 14,
              }}
            >
              {error}
            </div>
          ) : null}
          {busy ? (
            <div
              aria-live="polite"
              style={{
                marginBottom: 12,
                padding: "8px 12px",
                borderRadius: 6,
                background: "#172554",
                color: "#bfdbfe",
                fontSize: 13,
              }}
            >
              {status === "segmenting"
                ? "Segmenting…"
                : status === "grounding"
                  ? "Finding object…"
                  : "Generating result…"}
            </div>
          ) : null}

          <EditorCanvas
            imageUrl={imageUrl}
            imageSize={imageSize}
            mask={mask}
            tool={tool}
            brushRadius={brushRadius}
            disabled={busy || !imageUrl}
            onPointSelect={handlePointSelect}
            onBrushStroke={handleBrushStroke}
          />

          <ResultPanel
            originalUrl={imageUrl}
            maskPreviewUrl={maskPreview}
            resultUrl={resultUrl}
            latencyMs={latencyMs}
          />
        </main>

        <ControlPanel
          tool={tool}
          backend={backend}
          brushRadius={brushRadius}
          busy={busy}
          canGenerate={Boolean(imageFile && mask && maskHasInpaint(mask))}
          textPrompt={textPrompt}
          detections={detections}
          detectionIndex={detectionIndex}
          onUpload={handleUpload}
          onToolChange={setTool}
          onBrushRadiusChange={setBrushRadius}
          onClearMask={handleClearMask}
          onGenerate={handleGenerate}
          onTextPromptChange={setTextPrompt}
          onFindObject={handleFindObject}
          onDetectionIndexChange={handleDetectionIndexChange}
        />
      </div>
    </div>
  );
}
