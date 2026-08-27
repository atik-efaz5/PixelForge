"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { editByInstruction, inpaint, segment, selectByText, selectSmart } from "@/lib/api";
import { fitViewport, zoomIn, zoomOut, type CanvasViewport } from "@/lib/canvasView";
import { deriveEditorPhase, PHASE_LABELS } from "@/lib/editorPhase";
import { toUserFacingError, type UserFacingError } from "@/lib/formatError";
import {
  formatInpaintLine,
  formatSelectionLine,
  formatSmartSelectionLine,
  type ModelLine,
} from "@/lib/modelLabels";
import { validateImageFile } from "@/lib/imageUpload";
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
  InpaintCandidateResult,
  SelectionMode,
} from "@/types/api";
import { ControlPanel } from "@/components/ControlPanel";
import { CandidatePicker } from "@/components/CandidatePicker";
import { EditorCanvas } from "@/components/EditorCanvas";
import { ErrorBanner } from "@/components/ErrorBanner";
import { GenerationStatus } from "@/components/GenerationStatus";
import { HistoryPanel } from "@/components/HistoryPanel";
import { ModelStatusPanel } from "@/components/ModelStatusPanel";
import { ResultPanel } from "@/components/ResultPanel";

function revokeIfObjectUrl(url: string | null) {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

function extractRouting(
  metadata: Record<string, unknown> | undefined
): { model?: string; backend?: string } | undefined {
  const routing = metadata?.routing;
  if (!routing || typeof routing !== "object") return undefined;
  const record = routing as Record<string, unknown>;
  return {
    model: typeof record.model === "string" ? record.model : undefined,
    backend: typeof record.backend === "string" ? record.backend : undefined,
  };
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
  const [generateTwoCandidates, setGenerateTwoCandidates] = useState(false);
  const [pendingCandidates, setPendingCandidates] = useState<InpaintCandidateResult[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [backend, setBackend] = useState<InpaintBackend>("auto");
  const [selectionMode, setSelectionMode] = useState<SelectionMode>("smart");
  const [textPrompt, setTextPrompt] = useState("");
  const [editInstruction, setEditInstruction] = useState("");
  const [detections, setDetections] = useState<DetectionInfo[]>([]);
  const [detectionIndex, setDetectionIndex] = useState(0);
  const [status, setStatus] = useState<EditorStatus>("idle");
  const [userError, setUserError] = useState<UserFacingError | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [viewport, setViewport] = useState<CanvasViewport>(fitViewport());
  const [selectionLine, setSelectionLine] = useState<ModelLine | null>(null);
  const [inpaintLine, setInpaintLine] = useState<ModelLine | null>(null);
  const [resultModel, setResultModel] = useState<string | undefined>();
  const [resultBackend, setResultBackend] = useState<string | undefined>();
  const [operationElapsedMs, setOperationElapsedMs] = useState(0);
  const [lastRetry, setLastRetry] = useState<(() => void) | null>(null);
  const maskRef = useRef<Uint8Array | null>(null);
  const aiMaskRef = useRef<Uint8Array | null>(null);
  const maskHistoryRef = useRef(new MaskEditHistory());
  const sessionHistoryRef = useRef(new EditSessionHistory());
  const strokeSnapshotRef = useRef<Uint8Array | null>(null);
  const lastSelectionMetaRef = useRef<SelectionMetadata | undefined>(undefined);

  const busy = status !== "idle";

  const editorPhase = deriveEditorPhase({
    status,
    error: userError?.message ?? null,
    hasMask: Boolean(mask && maskHasInpaint(mask)),
    hasPendingResult,
    tool,
  });
  const phaseLabel = PHASE_LABELS[editorPhase];

  const setError = useCallback((raw: unknown) => {
    setUserError(toUserFacingError(raw));
  }, []);

  const clearError = useCallback(() => {
    setUserError(null);
    setLastRetry(null);
  }, []);

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
      if (snapshot.inpaint?.candidates && snapshot.inpaint.candidates.length > 1) {
        setPendingCandidates(
          snapshot.inpaint.candidates.map((candidate) => ({
            id: candidate.id,
            blob: new Blob(),
            url: candidate.url,
            rank: candidate.rank ?? 0,
            score: candidate.score ?? 0,
            seed: null,
          }))
        );
        setSelectedCandidateId(
          snapshot.inpaint.selectedCandidateId ??
            snapshot.inpaint.candidates[0]?.id ??
            null
        );
      } else {
        setPendingCandidates([]);
        setSelectedCandidateId(null);
      }
      if (snapshot.inpaint) {
        setResultModel(snapshot.inpaint.model);
        setResultBackend(
          typeof snapshot.inpaint.backend === "string"
            ? snapshot.inpaint.backend
            : undefined
        );
        setInpaintLine(
          formatInpaintLine({
            requestedBackend: snapshot.inpaint.backend,
            model: snapshot.inpaint.model,
            backend: snapshot.inpaint.backend,
          })
        );
      }
      if (snapshot.selection) {
        setSelectionLine(
          formatSelectionLine({
            method: snapshot.selection.method,
            model:
              snapshot.selection.method === "text"
                ? "grounding_dino"
                : "sam2",
            segmentationModel: "sam2",
          })
        );
      }
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
      clearError();
      setTool("select");
      setTextPrompt("");
      setEditInstruction("");
      setDetections([]);
      setDetectionIndex(0);
      setShowMaskOverlay(true);
      setShowMaskOnly(false);
      setFeatherRadius(0);
      setViewport(fitViewport());
      setSelectionLine(null);
      setInpaintLine(null);
      setSelectionMode("smart");
      setResultModel(undefined);
      setResultBackend(undefined);
      const entry = sessionHistoryRef.current.reset(url);
      applySnapshot(entry);
    },
    [applySnapshot, clearError, imageUrl]
  );

  const handleNewSession = useCallback(() => {
    revokeIfObjectUrl(imageUrl);
    sessionHistoryRef.current.dispose();
    setImageFile(null);
    setImageUrl(null);
    setImageSize(null);
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
    clearError();
    setTool("select");
    setTextPrompt("");
    setEditInstruction("");
    setDetections([]);
    setDetectionIndex(0);
    setShowMaskOverlay(true);
    setShowMaskOnly(false);
    setFeatherRadius(0);
    setViewport(fitViewport());
    setSelectionLine(null);
    setInpaintLine(null);
    setSelectionMode("smart");
    setResultModel(undefined);
    setResultBackend(undefined);
    setSessionEntries([]);
    setSessionIndex(-1);
    setCanSessionUndo(false);
    setCanSessionRedo(false);
    setStatus("idle");
  }, [clearError, imageUrl]);

  const handleUpload = useCallback(
    async (file: File) => {
      setStatus("uploading");
      clearError();
      try {
        const validated = await validateImageFile(file);
        const url = URL.createObjectURL(validated.file);
        resetSession(validated.file, url, {
          width: validated.width,
          height: validated.height,
        });
      } catch (err) {
        setError(err);
      } finally {
        setStatus("idle");
      }
    },
    [clearError, resetSession]
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
    clearError();
    try {
      if (selectionMode === "text") {
        const { blob, metadata } = await selectByText(
          imageFile,
          prompt,
          detectionIndex
        );
        setDetections(metadata.detections || []);
        setDetectionIndex(metadata.detection_index ?? 0);
        setSelectionLine(
          formatSelectionLine({
            method: "text",
            model: metadata.model,
            groundingBackend: metadata.grounding_backend,
            segmentationModel: metadata.segmentation_model,
          })
        );
        await applyMaskFromBlob(blob, {
          method: "text",
          prompt,
          label: metadata.selected_label ?? undefined,
          detectionIndex: metadata.detection_index ?? undefined,
        });
      } else {
        const { blob, metadata } = await selectSmart(imageFile, {
          selectionMode,
          prompt,
          detectionIndex,
        });
        setDetections(metadata.detections || []);
        setDetectionIndex(metadata.detection_index ?? 0);
        setSelectionLine(
          formatSmartSelectionLine({
            selectionMode,
            method: metadata.method as "point" | "text",
            confidenceTier: metadata.confidence_tier,
          })
        );
        await applyMaskFromBlob(blob, {
          method: "text",
          prompt,
          label: metadata.selected_label ?? undefined,
          detectionIndex: metadata.detection_index ?? undefined,
        });
      }
    } catch (err) {
      setError(err);
      setLastRetry(() => () => {
        void handleFindObject();
      });
    } finally {
      setStatus("idle");
    }
  }, [
    applyMaskFromBlob,
    busy,
    clearError,
    detectionIndex,
    imageFile,
    imageSize,
    selectionMode,
    setError,
    textPrompt,
  ]);

  const handleDetectionIndexChange = useCallback(
    async (index: number) => {
      setDetectionIndex(index);
      if (!imageFile || !textPrompt.trim() || busy) return;
      setStatus("grounding");
      clearError();
      try {
        const prompt = textPrompt.trim();
        if (selectionMode === "text") {
          const { blob, metadata } = await selectByText(imageFile, prompt, index);
          setDetections(metadata.detections || []);
          setSelectionLine(
            formatSelectionLine({
              method: "text",
              model: metadata.model,
              groundingBackend: metadata.grounding_backend,
              segmentationModel: metadata.segmentation_model,
            })
          );
          await applyMaskFromBlob(blob, {
            method: "text",
            prompt,
            label: metadata.selected_label ?? undefined,
            detectionIndex: index,
          });
        } else {
          const { blob, metadata } = await selectSmart(imageFile, {
            selectionMode,
            prompt,
            detectionIndex: index,
          });
          setDetections(metadata.detections || []);
          setSelectionLine(
            formatSmartSelectionLine({
              selectionMode,
              method: metadata.method as "point" | "text",
              confidenceTier: metadata.confidence_tier,
            })
          );
          await applyMaskFromBlob(blob, {
            method: "text",
            prompt,
            label: metadata.selected_label ?? undefined,
            detectionIndex: index,
          });
        }
      } catch (err) {
        setError(err);
      } finally {
        setStatus("idle");
      }
    },
    [applyMaskFromBlob, busy, clearError, imageFile, selectionMode, setError, textPrompt]
  );

  const handlePointSelect = useCallback(
    async (x: number, y: number) => {
      if (!imageFile || !imageSize || busy || selectionMode === "text") return;
      setStatus("segmenting");
      clearError();
      try {
        if (selectionMode === "point") {
          const { blob, metadata } = await segment(imageFile, x, y);
          setSelectionLine(
            formatSelectionLine({
              method: "point",
              model: metadata.model,
              backend: metadata.backend,
            })
          );
          await applyMaskFromBlob(blob, { method: "point" });
        } else {
          const { blob, metadata } = await selectSmart(imageFile, {
            selectionMode,
            x,
            y,
          });
          setSelectionLine(
            formatSmartSelectionLine({
              selectionMode,
              method: metadata.method as "point" | "text",
              confidenceTier: metadata.confidence_tier,
            })
          );
          await applyMaskFromBlob(blob, { method: "point" });
        }
      } catch (err) {
        setError(err);
      } finally {
        setStatus("idle");
      }
    },
    [applyMaskFromBlob, busy, clearError, imageFile, imageSize, selectionMode, setError]
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
    clearError();
    try {
      const maskBlob = await encodeMaskPng(
        maskRef.current,
        imageSize.width,
        imageSize.height
      );
      const candidateCount = generateTwoCandidates ? 2 : 1;
      const response = await inpaint(imageFile, maskBlob, backend, { candidateCount });

      if (response.mode === "single") {
        setPendingCandidates([]);
        setSelectedCandidateId(null);
        const routing = extractRouting(response.metadata.metadata);
        const url = URL.createObjectURL(response.blob);
        setResultModel(response.metadata.model);
        setResultBackend(response.metadata.backend);
        setInpaintLine(
          formatInpaintLine({
            requestedBackend: backend,
            model: response.metadata.model,
            backend: response.metadata.backend,
            routing,
          })
        );
        recordSession({
          operation: "INPAINT",
          label: "Inpainting complete",
          mask: maskRef.current,
          aiMask: aiMaskRef.current,
          resultUrl: url,
          selection: lastSelectionMetaRef.current,
          inpaint: {
            backend,
            model: response.metadata.model,
            latencyMs: response.metadata.latency_ms ?? undefined,
            candidateCount: 1,
          },
        });
        return;
      }

      const { metadata, candidates } = response.response;
      const routing = extractRouting(metadata.metadata);
      const recommended =
        candidates.find((c) => c.id === metadata.selected_candidate_id) ??
        candidates[0];
      setPendingCandidates(candidates);
      setSelectedCandidateId(recommended.id);
      setResultModel(metadata.model);
      setResultBackend(metadata.backend);
      setInpaintLine(
        formatInpaintLine({
          requestedBackend: backend,
          model: metadata.model,
          backend: metadata.backend,
          routing,
        })
      );
      recordSession({
        operation: "INPAINT",
        label: "Inpainting complete (2 candidates)",
        mask: maskRef.current,
        aiMask: aiMaskRef.current,
        resultUrl: recommended.url,
        selection: lastSelectionMetaRef.current,
        inpaint: {
          backend,
          model: metadata.model,
          latencyMs: metadata.latency_ms ?? undefined,
          candidateCount: 2,
          selectedCandidateId: recommended.id,
          ranking: metadata.ranking,
          candidates: candidates.map((candidate) => ({
            id: candidate.id,
            url: candidate.url,
            score: candidate.score,
            rank: candidate.rank,
          })),
        },
      });
    } catch (err) {
      setError(err);
      setLastRetry(() => () => {
        void handleGenerate();
      });
    } finally {
      setStatus("idle");
    }
  }, [
    backend,
    busy,
    clearError,
    generateTwoCandidates,
    imageFile,
    imageSize,
    recordSession,
    setError,
  ]);

  const handleUseResult = useCallback(() => {
    if (!hasPendingResult || !resultUrl) return;
    const selectedId =
      selectedCandidateId ??
      pendingCandidates.find((candidate) => candidate.url === resultUrl)?.id;
    const currentInpaint = sessionHistoryRef.current.current()?.inpaint;
    sessionHistoryRef.current.releaseCandidatesExcept(resultUrl);
    setPendingCandidates([]);
    setSelectedCandidateId(null);
    recordSession({
      operation: "RESULT_ACCEPTED",
      label: selectedId ? `Result accepted (${selectedId})` : "Result accepted",
      mask: maskRef.current,
      aiMask: aiMaskRef.current,
      resultUrl,
      selection: lastSelectionMetaRef.current,
      inpaint: {
        backend: currentInpaint?.backend ?? backend,
        model: currentInpaint?.model,
        latencyMs: currentInpaint?.latencyMs,
        selectedCandidateId: selectedId ?? undefined,
        candidateCount: pendingCandidates.length > 1 ? 2 : 1,
      },
    });
    setHasPendingResult(false);
  }, [
    backend,
    hasPendingResult,
    pendingCandidates,
    recordSession,
    resultUrl,
    selectedCandidateId,
  ]);

  const handleDiscardResult = useCallback(() => {
    const current = sessionHistoryRef.current.current();
    if (!current || current.operation !== "INPAINT" || !current.resultUrl) return;
    sessionHistoryRef.current.releaseResult(current.resultUrl);
    sessionHistoryRef.current.releaseCandidatesExcept(null);
    setPendingCandidates([]);
    setSelectedCandidateId(null);
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

  const handleSelectCandidate = useCallback((candidateId: string) => {
    const selected = pendingCandidates.find((candidate) => candidate.id === candidateId);
    if (!selected) return;
    setSelectedCandidateId(candidateId);
    setResultUrl(selected.url);
    const current = sessionHistoryRef.current.current();
    if (current?.operation === "INPAINT") {
      current.resultUrl = selected.url;
      if (current.inpaint) {
        current.inpaint = {
          ...current.inpaint,
          selectedCandidateId: candidateId,
        };
      }
    }
  }, [pendingCandidates]);

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
    clearError();
    try {
      const { blob, metadata } = await editByInstruction(imageFile, instruction);
      const url = URL.createObjectURL(blob);
      setResultModel(metadata.model);
      setResultBackend(metadata.backend);
      setInpaintLine(
        formatInpaintLine({
          requestedBackend: "instruct_pix2pix",
          model: metadata.model,
          backend: metadata.backend,
        })
      );
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
      setError(err);
    } finally {
      setStatus("idle");
    }
  }, [busy, clearError, editInstruction, imageFile, recordSession, setError]);

  useEffect(() => {
    if (mask && imageSize) {
      updateMaskPreview(mask, imageSize);
    }
  }, [featherRadius, imageSize, mask, updateMaskPreview]);

  useEffect(() => {
    if (!busy) {
      setOperationElapsedMs(0);
      return;
    }
    const started = Date.now();
    setOperationElapsedMs(0);
    const timer = window.setInterval(() => {
      setOperationElapsedMs(Date.now() - started);
    }, 100);
    return () => window.clearInterval(timer);
  }, [busy, status]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      ) {
        return;
      }

      const mod = event.metaKey || event.ctrlKey;
      if (mod && event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        if (canUndo) handleUndo();
        else if (canSessionUndo) handleSessionUndo();
        return;
      }
      if (mod && event.key.toLowerCase() === "z" && event.shiftKey) {
        event.preventDefault();
        if (canRedo) handleRedo();
        else if (canSessionRedo) handleSessionRedo();
        return;
      }

      if (busy) return;

      if (event.key === "b" || event.key === "B") {
        setTool("brush");
        return;
      }
      if (event.key === "e" || event.key === "E") {
        setTool("erase");
        return;
      }
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        setViewport((current) => zoomIn(current));
        return;
      }
      if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        setViewport((current) => zoomOut(current));
        return;
      }
      if (event.key === "0") {
        event.preventDefault();
        setViewport(fitViewport());
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    busy,
    canRedo,
    canSessionRedo,
    canSessionUndo,
    canUndo,
    handleRedo,
    handleSessionRedo,
    handleSessionUndo,
    handleUndo,
  ]);

  // Revoke replaced blob URLs when imageUrl changes or the editor unmounts.
  // Do not dispose the edit session here — resetSession owns session lifecycle.
  useEffect(() => {
    const url = imageUrl;
    return () => {
      revokeIfObjectUrl(url);
    };
  }, [imageUrl]);

  useEffect(() => {
    return () => {
      sessionHistoryRef.current.dispose();
    };
  }, []);

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

      <div className="editor-layout" style={{ flex: 1, display: "flex", minHeight: 0 }}>
        <main className="editor-main" style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, padding: 20, gap: 0 }}>
          {userError ? (
            <ErrorBanner
              error={userError}
              onDismiss={clearError}
              onRetry={lastRetry ?? undefined}
            />
          ) : null}
          <div
            aria-live="polite"
            style={{
              marginBottom: 12,
              padding: "8px 12px",
              borderRadius: 6,
              background: editorPhase === "error" ? "#3f1d1d" : "#1a1d24",
              border: "1px solid #2a2f3a",
              color: editorPhase === "error" ? "#fecaca" : "#9aa3b2",
              fontSize: 13,
            }}
          >
            {phaseLabel}
          </div>

          <ModelStatusPanel selection={selectionLine} inpainting={inpaintLine} />

          {busy ? (
            <GenerationStatus
              status={status}
              backend={status === "generating" ? backend : undefined}
              model={resultModel}
              elapsedMs={operationElapsedMs}
            />
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
            viewport={viewport}
            onViewportChange={setViewport}
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

          {pendingCandidates.length > 1 ? (
            <CandidatePicker
              options={pendingCandidates.map((candidate, index) => ({
                id: candidate.id,
                url: candidate.url,
                label: `Option ${index + 1}`,
                score: candidate.score,
                selected: candidate.id === selectedCandidateId,
              }))}
              disabled={busy}
              onSelect={handleSelectCandidate}
            />
          ) : null}

          <ResultPanel
            originalUrl={imageUrl}
            maskPreviewUrl={maskPreview}
            resultUrl={resultUrl}
            latencyMs={latencyMs}
            maskLabel={
              featherRadius > 0 ? "Edited mask (feather preview)" : "Edited mask"
            }
            hasAiMask={hasAiMask}
            resultModel={resultModel}
            resultBackend={resultBackend}
            requestedBackend={backend}
          />
        </main>

        <ControlPanel
          hasImage={Boolean(imageUrl)}
          onNewSession={handleNewSession}
          selectionMode={selectionMode}
          onSelectionModeChange={setSelectionMode}
          tool={tool}
          backend={backend}
          onBackendChange={setBackend}
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
          generateTwoCandidates={generateTwoCandidates}
          onGenerateTwoCandidatesChange={setGenerateTwoCandidates}
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
