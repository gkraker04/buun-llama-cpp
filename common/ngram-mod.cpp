#include "ngram-mod.h"

#include <algorithm>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);
    key_hashes.resize(size);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
    key_hashes[i] = full_hash(tokens);
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    if (key_hashes[i] == 0) {
        return EMPTY;
    }

    if (key_hashes[i] != full_hash(tokens)) {
        return EMPTY; // hash mismatch - collision detected
    }

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    std::fill(key_hashes.begin(), key_hashes.end(), 0);
    used = 0;
}

size_t common_ngram_mod::full_hash(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    // avoid 0 sentinel collision — use 1 if hash == 0
    if (res == 0) {
        res = 1;
    }

    return res;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * (sizeof(entries[0]) + sizeof(key_hashes[0]));
}

// binary format:
//   magic "NGMD" (4 bytes)
//   version uint32_t = 2
//   n       uint16_t
//   size    uint64_t
//   entries[size] int32_t
//   key_hashes[size] uint64_t

static constexpr const char * NGMOD_MAGIC = "NGMD";

// Helper to create parent directories recursively
static void create_parent_dirs(const std::string & filepath) {
    size_t pos = 0;
    while ((pos = filepath.find_first_of("/\\", pos + 1)) != std::string::npos) {
        std::string dir = filepath.substr(0, pos);
        if (!dir.empty()) {
            mkdir(dir.c_str(), 0755);
        }
    }
}

bool common_ngram_mod::save(const std::string & filename) const {
    create_parent_dirs(filename);
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    uint32_t version = 2;
    uint32_t n_save = static_cast<uint32_t>(n);
    uint64_t table_size = static_cast<uint64_t>(entries.size());

    out.write(NGMOD_MAGIC, 4);
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    out.write(reinterpret_cast<const char *>(&n_save), sizeof(n_save));
    out.write(reinterpret_cast<const char *>(&table_size), sizeof(table_size));
    out.write(reinterpret_cast<const char *>(entries.data()), table_size * sizeof(entry_t));
    out.write(reinterpret_cast<const char *>(key_hashes.data()), table_size * sizeof(key_hashes[0]));

    return out.good();
}

bool common_ngram_mod::load(const std::string & filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        return false;
    }

    char magic[4];
    fin.read(magic, 4);
    if (std::memcmp(magic, NGMOD_MAGIC, 4) != 0) {
        return false;
    }

    uint32_t version;
    fin.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1 && version != 2) {
        return false;
    }

    uint32_t n_loaded;
    fin.read(reinterpret_cast<char *>(&n_loaded), sizeof(n_loaded));

    uint64_t table_size;
    fin.read(reinterpret_cast<char *>(&table_size), sizeof(table_size));

    if (table_size != entries.size()) {
        fprintf(stderr, "%s: table_size mismatch: file=%llu, expected=%zu\n",
                __func__, (unsigned long long)table_size, entries.size());
        return false;
    }

    if (n_loaded != n) {
        fprintf(stderr, "%s: n mismatch: file=%u, current=%u (cache may be from different config)\n",
                __func__, n_loaded, (unsigned int)n);
        return false;
    }

    fin.read(reinterpret_cast<char *>(entries.data()), table_size * sizeof(entry_t));
    if (!fin.good()) {
        return false;
    }

    if (version == 2) {
        fin.read(reinterpret_cast<char *>(key_hashes.data()), table_size * sizeof(key_hashes[0]));
        if (!fin.good()) {
            return false;
        }
    } else {
        // version 1: no key_hashes in file — all key_hashes will be 0 (EMPTY),
        // so the cache effectively starts empty and rebuilds on the fly
        std::fill(key_hashes.begin(), key_hashes.end(), 0);
        std::fill(entries.begin(), entries.end(), EMPTY);
    }

    // recompute used count
    used = 0;
    for (const auto & e : entries) {
        if (e != EMPTY) {
            used++;
        }
    }

    return true;
}
