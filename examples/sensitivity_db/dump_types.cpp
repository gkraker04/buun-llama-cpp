// dump_types.cpp — print per-class type distribution in a GGUF
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gguf> [--list]\n", argv[0]); return 1; }
    bool list_mode = argc >= 3 && strcmp(argv[2], "--list") == 0;
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(argv[1], params);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }

    const int n = gguf_get_n_tensors(ctx);

    if (list_mode) {
        for (int i = 0; i < n; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            printf("%s\n", name);
        }
        gguf_free(ctx);
        return 0;
    }

    std::map<std::string, std::map<std::string, int>> class_types;
    for (int i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type t = gguf_get_tensor_type(ctx, i);
        std::string cls = "other";
        std::string s = name;
        if      (s == "token_embd.weight") cls = "token_embd";
        else if (s == "output.weight")     cls = "output";
        else if (s.find(".ffn_gate.") != std::string::npos) cls = "ffn_gate";
        else if (s.find(".ffn_up.")   != std::string::npos) cls = "ffn_up";
        else if (s.find(".ffn_down.") != std::string::npos) cls = "ffn_down";
        else if (s.find(".ssm_alpha.") != std::string::npos) cls = "ssm_alpha";
        else if (s.find(".ssm_beta.")  != std::string::npos) cls = "ssm_beta";
        else if (s.find(".ssm_out.")   != std::string::npos) cls = "ssm_out";
        else if (s.find(".attn_qkv.")  != std::string::npos) cls = "attn_qkv";
        else if (s.find(".attn_gate.") != std::string::npos) cls = "attn_gate";
        else if (s.find(".attn_output.") != std::string::npos) cls = "attn_output";
        else if (s.find(".attn_q.")    != std::string::npos) cls = "attn_q";
        else if (s.find(".attn_k.")    != std::string::npos) cls = "attn_k";
        else if (s.find(".attn_v.")    != std::string::npos) cls = "attn_v";
        class_types[cls][ggml_type_name(t)]++;
    }

    for (auto & kv : class_types) {
        printf("%-14s", kv.first.c_str());
        for (auto & tv : kv.second) printf(" %s:%d", tv.first.c_str(), tv.second);
        printf("\n");
    }

    gguf_free(ctx);
    return 0;
}
