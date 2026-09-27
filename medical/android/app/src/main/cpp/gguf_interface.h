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

typedef void (*TokenCallback)(const char* token, int32_t is_finished);

// Renamed to avoid collision with llama.cpp's own gguf_* symbols (gguf.h)
GGUF_EXPORT void* llm_init(const char* model_path, int32_t n_threads, int32_t n_ctx);

GGUF_EXPORT void llm_free(void* ctx);

GGUF_EXPORT int32_t llm_inference(
    void* ctx,
    const char* prompt,
    const char* system_prompt,
    float temperature,
    TokenCallback callback
);

GGUF_EXPORT void llm_stop(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // GGUF_INTERFACE_H
