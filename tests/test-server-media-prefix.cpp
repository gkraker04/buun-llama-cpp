#include "server-common.h"
#include "mtmd.h"

#include <cstdio>
#include <vector>

// Reproduces upstream U3: server_tokens::get_common_prefix must NOT treat two empty-id media
// chunks (unidentified video frames) as a cache match. Before the fix the id compare was
// content-independent for empty ids, so "" == "" plus equal token counts extended the common
// prefix past the media, letting one video's KV/state serve a different video of the same shape.
// The fix requires a non-empty, equal id (fail closed). No model or mmproj needed.

static server_tokens make(const char * img_id, size_t img_ntok,
                          const std::vector<llama_token> & lead) {
    server_tokens st(llama_tokens{}, /*has_mtmd=*/true);
    for (llama_token t : lead) {
        st.push_back(t);
    }
    mtmd_input_chunk * ch = mtmd_test_create_image_chunk(img_id, img_ntok);
    st.push_back(ch); // copies the chunk into the media map
    mtmd_input_chunk_free(ch);
    return st;
}

static int check(const char * name, size_t got, size_t want) {
    printf("%-26s common_prefix=%zu (want %zu) %s\n", name, got, want,
           got == want ? "OK" : "FAIL");
    return got == want ? 0 : 1;
}

static int check_identity(
        const char * name,
        server_tokens & tokens,
        int64_t n_tokens,
        bool want_valid,
        std::string * out = nullptr) {
    std::string identity;
    const bool valid =
        tokens.media_content_identity(n_tokens, identity);
    printf("%-26s identity_valid=%d (want %d) %s\n",
           name, valid, want_valid,
           valid == want_valid ? "OK" : "FAIL");
    if (out != nullptr) {
        *out = std::move(identity);
    }
    return valid == want_valid ? 0 : 1;
}

int main() {
    const std::vector<llama_token> lead = { 10, 11 }; // 2 shared leading text tokens
    int fails = 0;

    // U3 repro: two empty-id ("video frame") chunks of identical shape must diverge AT the media
    // (prefix = 2), never past it. HEAD/fixed = 2; parent/buggy = 5 (crosses). This is the red/green.
    {
        server_tokens a = make("", 3, lead);
        server_tokens b = make("", 3, lead);
        fails += check("empty-id video crossing", a.get_common_prefix(b), 2);
    }
    // Control: identical content id => legitimate image cache hit still extends past the media.
    {
        server_tokens a = make("sha:abc", 3, lead);
        server_tokens b = make("sha:abc", 3, lead);
        fails += check("same content-id match", a.get_common_prefix(b), 5);
    }
    // Control: different content ids diverge at the media (behavior unchanged by the fix).
    {
        server_tokens a = make("sha:abc", 3, lead);
        server_tokens b = make("sha:xyz", 3, lead);
        fails += check("diff content-id diverge", a.get_common_prefix(b), 2);
    }
    // Frontier records must reject the same unidentified/partial media cases,
    // while identical content yields an exact canonical comparison key.
    {
        server_tokens a = make("", 3, lead);
        fails += check_identity("empty-id identity", a, 5, false);
    }
    {
        server_tokens a = make("sha:abc", 3, lead);
        fails += check_identity("partial-media identity", a, 3, false);
    }
    {
        server_tokens a = make("sha:abc", 3, lead);
        server_tokens b = make("sha:abc", 3, lead);
        server_tokens c = make("sha:xyz", 3, lead);
        std::string ia;
        std::string ib;
        std::string ic;
        fails += check_identity("same-id identity A", a, 5, true, &ia);
        fails += check_identity("same-id identity B", b, 5, true, &ib);
        fails += check_identity("diff-id identity", c, 5, true, &ic);
        fails += check("identity equality", ia == ib, true);
        fails += check("identity distinction", ia != ic, true);
    }

    if (fails) {
        printf("FAILED (%d) — empty-id media crossing not fail-closed\n", fails);
        return 1;
    }
    printf("PASS: empty-id media fails closed; content-id caching intact\n");
    return 0;
}
