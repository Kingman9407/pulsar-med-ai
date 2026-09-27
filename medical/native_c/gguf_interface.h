/**
 * GGUF Native C++ Backend Interface for Flutter
 * 
 * This header defines the C-ABI expected by the Flutter Dart FFI pipeline
 * implemented in `lib/core/llm/gguf_bindings.dart`.
 * 
 * When implementing your C++ backend (e.g. using llama.cpp):
 * 1. Implement each function below with C linkage (`extern "C"`).
 * 2. Compile to a shared library:
 *    - Android: `libgguf_backend.so` (place in android/app/src/main/jniLibs/<arch>/)
 *    - iOS/macOS: `libgguf_backend.dylib` or statically linked framework
 *    - Windows: `gguf_backend.dll`
 *    - Linux: `libgguf_backend.so`
 */

#ifndef GGUF_INTERFACE_H
#define GGUF_INTERFACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define GGUF_EXPORT __declspec(dllexport)
#else
  #define GGUF_EXPORT __attribute__((visibility("default")))
#endif

/**
 * Token streaming callback signature
 * 
 * @param token Null-terminated UTF-8 string containing the newly generated token/piece.
 * @param is_finished Flag indicating if generation is complete (1 = EOS/finished, 0 = in progress).
 */
typedef void (*TokenCallback)(const char* token, int32_t is_finished);

/**
 * Initializes the GGUF model and inference context.
 * 
 * @param model_path File path to the .gguf model on the local device.
 * @param n_threads Number of CPU threads to utilize.
 * @param n_ctx Maximum context window size (e.g., 2048, 4096).
 * @return Opaque pointer to the initialized engine context (or NULL on failure).
 */
GGUF_EXPORT void* llm_init(const char* model_path, int32_t n_threads, int32_t n_ctx);

/**
 * Releases all resources, model weights, and context allocations.
 * 
 * @param ctx Pointer returned by gguf_init.
 */
GGUF_EXPORT void llm_free(void* ctx);

/**
 * Executes inference on the given prompt, invoking the callback for each token.
 * 
 * @param ctx Pointer returned by gguf_init.
 * @param prompt User query / message to evaluate.
 * @param system_prompt Persona or instruction framing.
 * @param temperature Sampling temperature (e.g., 0.7).
 * @param callback Function pointer invoked for each generated token.
 * @return 0 on success, non-zero error code on failure.
 */
GGUF_EXPORT int32_t llm_inference(
    void* ctx,
    const char* prompt,
    const char* system_prompt,
    float temperature,
    TokenCallback callback
);

/**
 * Requests premature cancellation/abortion of the ongoing inference.
 * 
 * @param ctx Pointer returned by gguf_init.
 */
GGUF_EXPORT void llm_stop(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // GGUF_INTERFACE_H
