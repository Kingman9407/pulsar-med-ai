#include "gguf_interface.h"

#include "llama.h"

#include <android/log.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define TAG "GgufNativeBackend"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// One context per load — keeps state for repeated inference calls
struct GgufHandle {
    llama_model*   model{nullptr};
    llama_context* ctx{nullptr};
    int32_t        n_ctx{2048};
    std::atomic<bool> stop_flag{false};
    std::mutex inference_mutex;
};

extern "C" {

static void llama_android_log_callback(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (!text || text[0] == '\0') return;
    int android_level = ANDROID_LOG_INFO;
    if (level == GGML_LOG_LEVEL_ERROR) {
        android_level = ANDROID_LOG_ERROR;
    } else if (level == GGML_LOG_LEVEL_WARN) {
        android_level = ANDROID_LOG_WARN;
    } else if (level == GGML_LOG_LEVEL_DEBUG) {
        android_level = ANDROID_LOG_DEBUG;
    }
    __android_log_print(android_level, TAG, "%s", text);
}

// ---------------------------------------------------------------------------
// gguf_init — load model from disk and create llama context
// ---------------------------------------------------------------------------
GGUF_EXPORT void* llm_init(const char* model_path,
                            int32_t    n_threads,
                            int32_t    n_ctx) {
    if (!model_path || model_path[0] == '\0') {
        LOGE("gguf_init: null or empty model_path");
        return nullptr;
    }

    llama_log_set(llama_android_log_callback, nullptr);

    LOGI("gguf_init: loading model from %s (threads=%d, n_ctx=%d)",
         model_path, n_threads, n_ctx);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU-only on Android

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        LOGE("gguf_init: llama_model_load_from_file failed for %s", model_path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (n_ctx    > 0) ? (uint32_t)n_ctx    : 2048u;
    cparams.n_threads = (n_threads> 0) ? (uint32_t)n_threads: 4u;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("gguf_init: llama_init_from_model failed");
        llama_model_free(model);
        return nullptr;
    }

    auto* h = new GgufHandle();
    h->model = model;
    h->ctx   = ctx;
    h->n_ctx = cparams.n_ctx;

    const llama_vocab* v = llama_model_get_vocab(model);
    LOGI("gguf_init: model loaded successfully (vocab=%d, n_ctx=%d)",
         v ? llama_vocab_n_tokens(v) : -1, h->n_ctx);
    return h;
}

// ---------------------------------------------------------------------------
// gguf_free — release model and context
// ---------------------------------------------------------------------------
GGUF_EXPORT void llm_free(void* handle) {
    if (!handle) return;
    auto* h = static_cast<GgufHandle*>(handle);
    h->stop_flag.store(true);
    if (h->ctx)   { llama_free(h->ctx);        h->ctx   = nullptr; }
    if (h->model) { llama_model_free(h->model); h->model = nullptr; }
    delete h;
    LOGI("gguf_free: resources released");
}

// ---------------------------------------------------------------------------
// gguf_stop — signal running inference to abort
// ---------------------------------------------------------------------------
GGUF_EXPORT void llm_stop(void* handle) {
    if (!handle) return;
    static_cast<GgufHandle*>(handle)->stop_flag.store(true);
    LOGI("gguf_stop: stop signal sent");
}

// ---------------------------------------------------------------------------
// gguf_inference — tokenise prompt, run the model, stream each token piece
// ---------------------------------------------------------------------------
GGUF_EXPORT int32_t llm_inference(void*         handle,
                                   const char*   prompt,
                                   const char*   system_prompt,
                                   float         temperature,
                                   TokenCallback callback) {
    if (!handle || !callback) {
        LOGE("gguf_inference: invalid arguments");
        return -1;
    }

    auto* h = static_cast<GgufHandle*>(handle);
    std::lock_guard<std::mutex> lock(h->inference_mutex);
    h->stop_flag.store(false);

    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    // ---- 1. Build ChatML prompt string ----
    std::string full_prompt;
    if (system_prompt && strlen(system_prompt) > 0) {
        full_prompt += "<|im_start|>system\n";
        full_prompt += system_prompt;
        full_prompt += "<|im_end|>\n";
    }
    full_prompt += "<|im_start|>user\n";
    full_prompt += (prompt ? prompt : "");
    full_prompt += "<|im_end|>\n<|im_start|>assistant\n";

    LOGI("gguf_inference: prompt len=%zu", full_prompt.size());

    // ---- 2. Tokenise ----
    // Overestimate token count for buffer
    const int prompt_len_est = static_cast<int>(full_prompt.size()) + 8;
    std::vector<llama_token> tokens(prompt_len_est);

    int n_tokens = llama_tokenize(
        vocab,
        full_prompt.c_str(),
        static_cast<int32_t>(full_prompt.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/true
    );

    if (n_tokens < 0) {
        // Buffer was too small, retry with exact size
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            vocab,
            full_prompt.c_str(),
            static_cast<int32_t>(full_prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            /*add_special=*/true,
            /*parse_special=*/true
        );
    }

    if (n_tokens < 0) {
        LOGE("gguf_inference: tokenization failed (code %d)", n_tokens);
        callback("[tokenization failed]", 1);
        return -2;
    }
    tokens.resize(n_tokens);

    LOGI("gguf_inference: %d prompt tokens", n_tokens);

    // ---- 3. Clear KV cache from previous turn ----
    llama_memory_t mem = llama_get_memory(h->ctx);
    if (mem) { llama_memory_clear(mem, /*data=*/true); }

    // ---- 4. Decode prompt tokens in one batch ----
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        if (llama_decode(h->ctx, batch) != 0) {
            LOGE("gguf_inference: llama_decode failed on prompt");
            callback("[decode error]", 1);
            return -3;
        }
    }

    // ---- 5. Set up sampler chain ----
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(
        (temperature > 0.0f && temperature < 5.0f) ? temperature : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ---- 6. Auto-regressive generation loop ----
    const int max_new_tokens = std::min(256, h->n_ctx - n_tokens);
    char piece_str[512];
    char check_buf[512];

    int n_generated = 0;
    while (n_generated < max_new_tokens) {
        if (h->stop_flag.load()) {
            LOGI("gguf_inference: aborted by stop signal at token %d", n_generated);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl, h->ctx, -1);

        // 1. Direct EOS / EOG / ChatML stop token check
        // SmolLM2 ChatML: 2 = <|im_end|>, 0 = <|endoftext|>, 1 = <|im_start|>
        if (id == 2 || id == 0 || llama_vocab_is_eog(vocab, id) || id == llama_vocab_eos(vocab)) {
            LOGI("gguf_inference: stop token ID %d reached at token %d", id, n_generated);
            break;
        }

        // 2. Check token text with special=true to catch any boundary markers
        int check_len = llama_token_to_piece(vocab, id, check_buf, sizeof(check_buf) - 1, 0, /*special=*/true);
        if (check_len > 0) {
            check_buf[check_len] = '\0';
            if (strstr(check_buf, "<|im_end|>") || strstr(check_buf, "<|endoftext|>") || strstr(check_buf, "<|im_start|>")) {
                LOGI("gguf_inference: boundary marker '%s' reached at token %d", check_buf, n_generated);
                break;
            }
        }

        // 3. Convert token id → UTF-8 piece for client callback
        int piece_len = llama_token_to_piece(
            vocab,
            id,
            piece_str,
            sizeof(piece_str) - 1,
            0,
            /*special=*/false
        );

        if (piece_len > 0) {
            piece_str[piece_len] = '\0';
            callback(piece_str, 0);
        }

        // Decode the new token to advance the KV cache
        llama_batch next = llama_batch_get_one(&const_cast<llama_token&>(id), 1);
        if (llama_decode(h->ctx, next) != 0) {
            LOGE("gguf_inference: llama_decode failed at token %d", n_generated);
            break;
        }

        n_generated++;
        if (n_generated % 10 == 0) {
            LOGI("gguf_inference: generated %d tokens so far...", n_generated);
        }
    }

    llama_sampler_free(smpl);
    callback("", 1); // Signal done
    LOGI("gguf_inference: generated %d tokens", n_generated);
    return 0;
}

} // extern "C"
