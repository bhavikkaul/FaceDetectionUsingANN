import ctypes
import numpy as np
import cv2
import sys
import os

# Load the shared library
lib_path = "/Users/bhavikkaul/Downloads/files/libfacedetection-master/build/libfacedetect_wrapper.dylib"
lib = ctypes.CDLL(lib_path)

# Set up function signature
lib.facedetect_cnn_c.restype = ctypes.POINTER(ctypes.c_int)
lib.facedetect_cnn_c.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

DETECT_BUFFER_SIZE = 0x20000

def detect_faces(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot load image: {image_path}")
        return

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    buffer = (ctypes.c_ubyte * DETECT_BUFFER_SIZE)()
    img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    import time
    t0 = time.time()
    results = lib.facedetect_cnn_c(buffer, img_data, image.shape[1], image.shape[0], int(image.strides[0]))
    elapsed = (time.time() - t0) * 1000

    num_faces = results[0] if results else 0
    print(f"Detection time: {elapsed:.1f}ms")
    print(f"Faces detected: {num_faces}")

    result_image = image.copy()

    # Access raw memory from results pointer as shorts
    # results is int*, starting at results[1] we have shorts (16 shorts per face)
    raw = ctypes.cast(results, ctypes.POINTER(ctypes.c_ubyte))
    # Skip first 4 bytes (the int count), then read short arrays
    shorts_start = 4  # bytes offset

    for i in range(num_faces):
        offset = shorts_start + i * 16 * 2  # 16 shorts * 2 bytes each
        face_shorts = (ctypes.c_short * 16).from_buffer_copy(
            bytes(raw[offset:offset+32])
        )
        confidence = face_shorts[0]
        x = face_shorts[1]
        y = face_shorts[2]
        w = face_shorts[3]
        h = face_shorts[4]
        landmarks = [(face_shorts[5 + j*2], face_shorts[5 + j*2 + 1]) for j in range(5)]

        print(f"  Face {i}: confidence={confidence}, bbox=[{x},{y},{w},{h}]")

        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, f"conf:{confidence}", (x, max(0, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        colors = [(255,0,0),(0,0,255),(0,255,0),(255,0,255),(0,255,255)]
        for j, (lx, ly) in enumerate(landmarks):
            cv2.circle(result_image, (lx, ly), 3, colors[j], -1)

    cv2.imwrite(output_path, result_image)
    print(f"\nResult saved to: {output_path}")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/libfacedetection-master/test.jpeg"
    out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/face_detection_result.jpg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    detect_faces(img, out)
