#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

    // persist the hash table to/from disk
    bool save(const std::string & filename) const;
    bool load(const std::string & filename);

private:
    size_t n; // ngram size to hash

    size_t used;

    size_t full_hash(const entry_t * tokens) const;

    std::vector<entry_t> entries;
    std::vector<size_t>  key_hashes; // full 64-bit hash of the ngram key, 0 = EMPTY
};
