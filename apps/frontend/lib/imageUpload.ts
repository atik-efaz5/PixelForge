/** Client-side image upload checks before sending to the API. */

export const MIN_IMAGE_DIMENSION = 8;
export const MAX_IMAGE_PIXELS = 16_777_216;

export interface ValidatedImage {
  file: File;
  width: number;
  height: number;
}

export function validateImageFile(file: File): Promise<ValidatedImage> {
  if (!file.size) {
    return Promise.reject(new Error("Image file is empty."));
  }
  if (!file.type.startsWith("image/")) {
    return Promise.reject(
      new Error("Unsupported file type. Use PNG, JPEG, or WebP.")
    );
  }

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const { naturalWidth: width, naturalHeight: height } = img;
      if (width < MIN_IMAGE_DIMENSION || height < MIN_IMAGE_DIMENSION) {
        reject(
          new Error(
            `Image is too small (${width}×${height}). Minimum dimension is ${MIN_IMAGE_DIMENSION}px.`
          )
        );
        return;
      }
      if (width * height > MAX_IMAGE_PIXELS) {
        reject(new Error("Image exceeds the maximum allowed pixel count."));
        return;
      }
      resolve({ file, width, height });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read the selected image. It may be corrupted."));
    };
    img.src = url;
  });
}
