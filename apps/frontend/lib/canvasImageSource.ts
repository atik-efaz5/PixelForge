/** Skip decoding when the canvas already shows this source URL. */
export function shouldReloadCanvasImage(
  loadedSourceUrl: string | null,
  sourceUrl: string,
  hasDecodedImage: boolean
): boolean {
  return !(loadedSourceUrl === sourceUrl && hasDecodedImage);
}
