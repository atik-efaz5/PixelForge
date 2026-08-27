"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { editByInstruction, inpaint, segment, selectByText } from "@/lib/api";
import {
  EditSessionHistory,
  type EditSessionSnapshot,
  type SelectionMetadata,
} from "@/lib/editSession";
import {
  MaskEditHistory,
  cloneMask,
  createEmptyMask,
  decodeMaskPng,
  dilateMask,
  encodeMaskPng,
  erodeMask,
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
import { HistoryPanel } from "@/components/HistoryPanel";
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
  const [eraserRadius, setEraserRadius] = useState(24);
  const [featherRadius, setFeatherRadius] = useState(0);
  const [morphAmount, setMorphAmount] = useState(1);
  const [showMaskOverlay, setShowMaskOverlay] = useState(true);
  const [showMaskOnly, setShowMaskOnly] = useState(false);
  const [hasAiMask, setHasAiMask] = useState(false);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [sessionEntries, setSessionEntries] = useState<readonly EditSessionSnapshot[]>([]);
  const [sessionIndex, setSessionIndex] = useState(-1);
  const [canSessionUndo, setCanSessionUndo] = useState(false);
  const [canSessionRedo, setCanSessionRedo] = useState(false);
  const [hasPendingResult, setHasPendingResult] = useState(false);
  const [backend] = useState<InpaintBackend>("moebius");
  const [textPrompt, setTextPrompt] = useState("");
  const [editInstruction, setEditInstruction] = useState("");
  const [detections, setDetections] = useState<DetectionInfo[]>([]);
  const [detectionIndex, setDetectionIndex] = useState(0);
  const [status, setStatus] = useState<EditorStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const maskRef = useRef<Uint8Array | null>(null);
  const aiMaskRef = useRef<Uint8Array | null>(null);
  const maskHistoryRef = useRef(new MaskEditHistory());
  const sessionHistoryRef = useRef(new EditSessionHistory());
  const strokeSnapshotRef = useRef<Uint8Array | null>(null);
  const lastSelectionMetaRef = useRef<SelectionMetadata | undefined>(undefined);

  const busy = status !== "idle";

  const syncMaskHistoryFlags = useCallback(() => {
    const history = maskHistoryRef.current;
    setCanUndo(history.canUndo());
    setCanRedo(history.canRedo());
  }, []);

  const syncSessionUi = useCallback(() => {
    const session = sessionHistoryRef.current;
    setSessionEntries(session.snapshotEntries);
    setSessionIndex(session.currentIndex);
    setCanSessionUndo(session.canUndo());
    setCanSessionRedo(session.canRedo());
    const current = session.current();
    setHasPendingResult(current?.operation === "INPAINT");
  }, []);

  const updateMaskPreview = useCallback(
    (nextMask: Uint8Array | null, size: ImageDimensions | null) => {
      if (!nextMask || !size) {
        setMaskPreview(null);
        return;
      }
      setMaskPreview(
        maskPreviewUrl(nextMask, size.width, size.height, {
          featherRadius: featherRadius > 0 ? featherRadius : undefined,
        })
      );
    },
    [featherRadius]
  );

  const applyMask = useCallback(
    (nextMask: Uint8Array | null) => {
      if (!imageSize) return;
      maskRef.current = nextMask;
      setMask(nextMask);
      if (nextMask) {
        updateMaskPreview(nextMask, imageSize);
      } else {
        setMaskPreview(null);
      }
      syncMaskHistoryFlags();
    },
    [imageSize, syncMaskHistoryFlags, updateMaskPreview]
  );

  const applySnapshot = useCallback(
    (snapshot: EditSessionSnapshot) => {
      if (imageSize) {
        if (snapshot.mask) {
          applyMask(cloneMask(snapshot.mask));
        } else {
          applyMask(null);
        }
      }
      aiMaskRef.current = snapshot.aiMask ? cloneMask(snapshot.aiMask) : null;
      setHasAiMask(Boolean(snapshot.aiMask));
      setResultUrl(snapshot.resultUrl);
      setLatencyMs(snapshot.inpaint?.latencyMs ?? null);
      if (snapshot.selection?.prompt) {
        setTextPrompt(snapshot.selection.prompt);
      }
      syncSessionUi();
    },
    [applyMask, imageSize, syncSessionUi]
  );

  const recordSession = useCallback(
    (
      input: Parameters<EditSessionHistory["append"]>[0],
      options?: { apply?: boolean }
    ) => {
      const entry = sessionHistoryRef.current.append(input);
      if (options?.apply !== false) {
        applySnapshot(entry);
      } else {
        syncSessionUi();
      }
      return entry;
    },
    [applySnapshot, syncSessionUi]
  );

  const commitMaskEdit = useCallback(
    (nextMask: Uint8Array, options?: { recordHistory?: boolean }) => {
      if (!maskRef.current || options?.recordHistory === false) {
        applyMask(nextMask);
        return;
      }
      maskHistoryRef.current.push(maskRef.current);
      applyMask(nextMask);
    },
    [applyMask]
  );

  const resetSession = useCallback(
    (file: File, url: string, size: ImageDimensions) => {
      revokeIfObjectUrl(imageUrl);
      sessionHistoryRef.current.dispose();
      setImageFile(file);
      setImageUrl(url);
      setImageSize(size);
      setMask(null);
      maskRef.current = null;
      aiMaskRef.current = null;
      maskHistoryRef.current.clear();
      lastSelectionMetaRef.current = undefined;
      setHasAiMask(false);
      setCanUndo(false);
      setCanRedo(false);
      setMaskPreview(null);
      setResultUrl(null);
      setLatencyMs(null);
      setHasPendingResult(false);
      setError(null);
      setTool("select");
      setTextPrompt("");
      setEditInstruction("");
      setDetections([]);
      setDetectionIndex(0);
      setShowMaskOverlay(true);
      setShowMaskOnly(false);
      setFeatherRadius(0);
      const entry = sessionHistoryRef.current.reset(url);
      applySnapshot(entry);
    },
    [applySnapshot, imageUrl]
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
    async (
      blob: Blob,
      selection: SelectionMetadata
    ) => {
      if (!imageSize) return;
      const decoded = await decodeMaskPng(blob, imageSize);
      aiMaskRef.current = cloneMask(decoded);
      maskHistoryRef.current.clear();
      lastSelectionMetaRef.current = selection;
      setHasAiMask(true);
      applyMask(decoded);
      recordSession({
        operation: "SELECT",
        label: "Object selected",
        mask: decoded,
        aiMask: decoded,
        selection,
      });
      setTool("brush");
    },
    [applyMask, imageSize, recordSession]
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
      await applyMaskFromBlob(blob, {
        method: "text",
        prompt,
        label: metadata.selected_label,
        detectionIndex: metadata.detection_index,
      });
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
        await applyMaskFromBlob(blob, {
          method: "text",
          prompt: textPrompt.trim(),
          label: metadata.selected_label,
          detectionIndex: index,
        });
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
        await applyMaskFromBlob(blob, { method: "point" });
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

  const recordMaskRefined = useCallback(
    (label = "Mask refined") => {
      if (!maskRef.current) return;
      recordSession({
        operation: "MASK_EDIT",
        label,
        mask: maskRef.current,
        aiMask: aiMaskRef.current,
        resultUrl,
        selection: lastSelectionMetaRef.current,
      });
    },
    [recordSession, resultUrl]
  );

  const handleStrokeStart = useCallback(() => {
    if (maskRef.current) {
      strokeSnapshotRef.current = cloneMask(maskRef.current);
    }
  }, []);

  const handleStrokeEnd = useCallback(() => {
    if (strokeSnapshotRef.current && maskRef.current) {
      maskHistoryRef.current.push(strokeSnapshotRef.current);
      strokeSnapshotRef.current = null;
      syncMaskHistoryFlags();
      recordMaskRefined();
    }
  }, [recordMaskRefined, syncMaskHistoryFlags]);

  const handleBrushStroke = useCallback(
    (x: number, y: number, mode: "brush" | "erase") => {
      if (!imageSize || !maskRef.current) return;
      const next = cloneMask(maskRef.current);
      const radius = mode === "erase" ? eraserRadius : brushRadius;
      paintBrush(next, imageSize.width, imageSize.height, x, y, radius, mode);
      maskRef.current = next;
      setMask(next);
      updateMaskPreview(next, imageSize);
    },
    [brushRadius, eraserRadius, imageSize, updateMaskPreview]
  );

  const handleExpandMask = useCallback(() => {
    if (!imageSize || !maskRef.current) return;
    const next = dilateMask(
      maskRef.current,
      imageSize.width,
      imageSize.height,
      morphAmount
    );
    commitMaskEdit(next, { recordHistory: true });
    recordMaskRefined("Mask expanded");
  }, [commitMaskEdit, imageSize, morphAmount, recordMaskRefined]);

  const handleShrinkMask = useCallback(() => {
    if (!imageSize || !maskRef.current) return;
    const next = erodeMask(
      maskRef.current,
      imageSize.width,
      imageSize.height,
      morphAmount
    );
    commitMaskEdit(next, { recordHistory: true });
    recordMaskRefined("Mask shrunk");
  }, [commitMaskEdit, imageSize, morphAmount, recordMaskRefined]);

  const handleUndo = useCallback(() => {
    if (!maskRef.current) return;
    const restored = maskHistoryRef.current.undo(maskRef.current);
    if (restored) applyMask(restored);
  }, [applyMask]);

  const handleRedo = useCallback(() => {
    if (!maskRef.current) return;
    const restored = maskHistoryRef.current.redo(maskRef.current);
    if (restored) applyMask(restored);
  }, [applyMask]);

  const handleResetToAiMask = useCallback(() => {
    if (!aiMaskRef.current) return;
    commitMaskEdit(cloneMask(aiMaskRef.current), { recordHistory: true });
    recordSession({
      operation: "MASK_RESET",
      label: "Reset to AI mask",
      mask: aiMaskRef.current,
      aiMask: aiMaskRef.current,
      resultUrl,
      selection: lastSelectionMetaRef.current,
    });
  }, [commitMaskEdit, recordSession, resultUrl]);

  const handleClearMask = useCallback(() => {
    if (!imageSize) return;
    const cleared = createEmptyMask(imageSize.width, imageSize.height);
    commitMaskEdit(cleared, { recordHistory: true });
    recordSession({
      operation: "MASK_EDIT",
      label: "Mask cleared",
      mask: cleared,
      aiMask: aiMaskRef.current,
      resultUrl,
      selection: lastSelectionMetaRef.current,
    });
  }, [commitMaskEdit, imageSize, recordSession, resultUrl]);

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
      const url = URL.createObjectURL(blob);
      recordSession({
        operation: "INPAINT",
        label: "Moebius generated",
        mask: maskRef.current,
        aiMask: aiMaskRef.current,
        resultUrl: url,
        selection: lastSelectionMetaRef.current,
        inpaint: {
          backend,
          model: metadata.model,
          latencyMs: metadata.latency_ms ?? undefined,
        },
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Inpainting failed.";
      setError(message);
    } finally {
      setStatus("idle");
    }
  }, [backend, busy, imageFile, imageSize, recordSession]);

  const handleUseResult = useCallback(() => {
    if (!hasPendingResult || !resultUrl) return;
    recordSession({
      operation: "RESULT_ACCEPTED",
      label: "Result accepted",
      mask: maskRef.current,
      aiMask: aiMaskRef.current,
      resultUrl,
      selection: lastSelectionMetaRef.current,
      inpaint: sessionHistoryRef.current.current()?.inpaint,
    });
    setHasPendingResult(false);
  }, [hasPendingResult, recordSession, resultUrl]);

  const handleDiscardResult = useCallback(() => {
    const current = sessionHistoryRef.current.current();
    if (!current || current.operation !== "INPAINT" || !current.resultUrl) return;
    sessionHistoryRef.current.releaseResult(current.resultUrl);
    const previous = sessionHistoryRef.current.previousResultSnapshot();
    recordSession({
      operation: "RESULT_REJECTED",
      label: "Result discarded",
      mask: maskRef.current,
      aiMask: aiMaskRef.current,
      resultUrl: previous?.resultUrl ?? null,
      selection: lastSelectionMetaRef.current,
    });
    setHasPendingResult(false);
  }, [recordSession]);

  const handleRestorePrevious = useCallback(() => {
    const previous = sessionHistoryRef.current.previousResultSnapshot();
    if (!previous) return;
    const idx = sessionHistoryRef.current.snapshotEntries.findIndex(
      (e) => e.id === previous.id
    );
    if (idx >= 0) {
      const snap = sessionHistoryRef.current.goTo(idx);
      if (snap) applySnapshot(snap);
    }
    setHasPendingResult(false);
  }, [applySnapshot]);

  const handleSessionUndo = useCallback(() => {
    const snap = sessionHistoryRef.current.undo();
    if (snap) applySnapshot(snap);
  }, [applySnapshot]);

  const handleSessionRedo = useCallback(() => {
    const snap = sessionHistoryRef.current.redo();
    if (snap) applySnapshot(snap);
  }, [applySnapshot]);

  const handleSelectHistoryEntry = useCallback(
    (index: number) => {
      const snap = sessionHistoryRef.current.goTo(index);
      if (snap) applySnapshot(snap);
    },
    [applySnapshot]
  );

  const handleApplyInstruction = useCallback(async () => {
    if (!imageFile || busy) return;
    const instruction = editInstruction.trim();
    if (!instruction) {
      setError("Enter an instruction to apply.");
      return;
    }
    setStatus("instruction_editing");
    setError(null);
    try {
      const { blob, metadata } = await editByInstruction(imageFile, instruction);
      const url = URL.createObjectURL(blob);
      recordSession({
        operation: "INPAINT",
        label: "Instruction edit",
        mask: maskRef.current,
        aiMask: aiMaskRef.current,
        resultUrl: url,
        inpaint: {
          backend: "instruct_pix2pix",
          model: metadata.model,
          latencyMs: metadata.latency_ms ?? undefined,
        },
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Instruction editing failed.";
      setError(message);
    } finally {
      setStatus("idle");
    }
  }, [busy, editInstruction, imageFile, recordSession]);

  useEffect(() => {
    if (mask && imageSize) {
      updateMaskPreview(mask, imageSize);
    }
  }, [featherRadius, imageSize, mask, updateMaskPreview]);

  useEffect(() => {
    return () => {
      revokeIfObjectUrl(imageUrl);
      sessionHistoryRef.current.dispose();
    };
  }, [imageUrl]);

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
          Select · refine mask · preview · generate with Moebius
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
                  : status === "instruction_editing"
                    ? "Applying instruction…"
                    : "Generating result…"}
            </div>
          ) : null}

          <EditorCanvas
            imageUrl={imageUrl}
            imageSize={imageSize}
            mask={mask}
            tool={tool}
            brushRadius={brushRadius}
            eraserRadius={eraserRadius}
            featherRadius={featherRadius}
            showOverlay={showMaskOverlay}
            showMaskOnly={showMaskOnly}
            disabled={busy || !imageUrl}
            onPointSelect={handlePointSelect}
            onBrushStroke={handleBrushStroke}
            onStrokeStart={handleStrokeStart}
            onStrokeEnd={handleStrokeEnd}
          />

          <HistoryPanel
            entries={sessionEntries}
            currentIndex={sessionIndex}
            canUndo={canSessionUndo}
            canRedo={canSessionRedo}
            busy={busy}
            hasPendingResult={hasPendingResult}
            onSelectEntry={handleSelectHistoryEntry}
            onSessionUndo={handleSessionUndo}
            onSessionRedo={handleSessionRedo}
            onUseResult={handleUseResult}
            onDiscardResult={handleDiscardResult}
            onRestorePrevious={handleRestorePrevious}
          />

          <ResultPanel
            originalUrl={imageUrl}
            maskPreviewUrl={maskPreview}
            resultUrl={resultUrl}
            latencyMs={latencyMs}
            maskLabel={
              featherRadius > 0 ? "Edited mask (feather preview)" : "Edited mask"
            }
            hasAiMask={hasAiMask}
          />
        </main>

        <ControlPanel
          tool={tool}
          backend={backend}
          brushRadius={brushRadius}
          eraserRadius={eraserRadius}
          morphAmount={morphAmount}
          featherRadius={featherRadius}
          showMaskOverlay={showMaskOverlay}
          showMaskOnly={showMaskOnly}
          hasAiMask={hasAiMask}
          canUndo={canUndo}
          canRedo={canRedo}
          busy={busy}
          canGenerate={Boolean(imageFile && mask && maskHasInpaint(mask))}
          textPrompt={textPrompt}
          detections={detections}
          detectionIndex={detectionIndex}
          onUpload={handleUpload}
          onToolChange={setTool}
          onBrushRadiusChange={setBrushRadius}
          onEraserRadiusChange={setEraserRadius}
          onMorphAmountChange={setMorphAmount}
          onFeatherRadiusChange={setFeatherRadius}
          onShowMaskOverlayChange={setShowMaskOverlay}
          onShowMaskOnlyChange={setShowMaskOnly}
          onExpandMask={handleExpandMask}
          onShrinkMask={handleShrinkMask}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onResetToAiMask={handleResetToAiMask}
          onClearMask={handleClearMask}
          onGenerate={handleGenerate}
          onTextPromptChange={setTextPrompt}
          onFindObject={handleFindObject}
          onDetectionIndexChange={handleDetectionIndexChange}
          editInstruction={editInstruction}
          canApplyInstruction={Boolean(imageFile && editInstruction.trim())}
          onEditInstructionChange={setEditInstruction}
          onApplyInstruction={handleApplyInstruction}
        />
      </div>
    </div>
  );
}
