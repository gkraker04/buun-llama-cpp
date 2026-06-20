# Decentralized Pipeline-Parallel Inference — Research Direction

**Created:** 2026-06-20  
**Status:** Early research / idea exploration  
**Foundation:** Shard-style pipeline-parallel RPC (experiments/shard-pipeline-parallel/)

---

## Vision

"BitTorrent for LLMs" — a decentralized network where participants share GPU resources to run large models collaboratively.

**Core concept:**
- One person runs an RPC server, shares their IP:port
- Others connect their GPUs as additional RPC servers
- Coordinator discovers all nodes and assigns layers optimally
- Participants get inference slots proportional to their contribution

**Why this could work:**
- Pipeline-parallel RPC + speculative decoding over WAN (what we're building) hides latency
- Model weights are static, large files — perfect for BitTorrent/IPFS distribution
- Each node loads only their assigned layers (or redundant layers for fault tolerance)
- Incentive: help run the big model, get a slot to use it

---

## Architecture

### Bootstrap (simple)
1. Node operator starts RPC server, shares connection info
2. Other operators connect their GPUs
3. Coordinator (could be first node, or separate process) discovers all nodes
4. Topology solver optimizes node ordering for minimum latency
5. Layer assignment distributes model across nodes to fit VRAM

### Layer Assignment
- Query each node's VRAM
- Distribute model layers across nodes (contiguous blocks per node)
- Optimize for latency using topology solver (from PLAN.md T4.2)
- Support redundant layers for fault tolerance

### Model Distribution
- Model weights distributed via BitTorrent/IPFS
- Each node downloads full model, loads only assigned layers
- Alternative: shard the model file itself (each node downloads only their layers)

### Incentive Mechanism
- Track contribution: uptime, VRAM provided, compute cycles
- Allocate inference slots proportionally
- No crypto needed for trusted federation — simple ledger
- Long-term: "torrent" is the model, network allows redundant layers so separate model instances can serve more of the network

---

## Research Questions

### 1. Verification
**Problem:** How do you know a node isn't returning garbage?

**Options:**
- **Redundant computation** — compute on 2 nodes, compare results. Doubles cost but simple.
- **Test prompts** — periodic prompts with known answers. Lightweight but gameable.
- **Cryptographic proofs** — zero-knowledge or verifiable compute. Complex but trustless.

**Recommendation:** Start with redundant computation for trusted federation. Graduate to cryptographic proofs for open network.

### 2. Fault Tolerance
**Problem:** What happens when a node drops?

**Options:**
- **Redundant layers** — multiple nodes host the same layers
- **Graceful degradation** — coordinator reroutes around failed nodes
- **Checkpointing** — periodic state snapshots for recovery

**Recommendation:** Redundant layers + graceful degradation. Coordinator detects node failure, reallocates layers to remaining nodes.

### 3. Consensus
**Problem:** Who decides layer assignments?

**Options:**
- **Centralized coordinator** — simple, single point of failure
- **Distributed consensus** — complex, but decentralized
- **Hybrid** — coordinator proposes, nodes vote

**Recommendation:** Start with centralized coordinator for simplicity. Research distributed consensus for later phases.

---

## Existing Attempts

- **Petals** (decentralized BLOOM) — tried this, struggled with latency and reliability. No speculative decoding, so WAN latency was a bottleneck.
- **Gensyn** — compute marketplace, but not LLM-specific
- **Various blockchain compute networks** — mostly vaporware or too slow for inference

**Our advantage:** Speculative decoding over WAN hides latency. This is the key innovation that could make it work.

---

## Research Roadmap

### Phase 1: Foundation (what we're building now)
- Prove pipeline-parallel RPC works over WAN
- Async RPC backend (T1.x)
- Coordinator + direct-return + n-gram draft (T2.x)
- MTP integration (T3.x)
- WAN optimization (T4.x)

**Outcome:** Working pipeline-parallel inference over WAN with speculative decoding.

### Phase 2: Node Discovery
- Simple HTTP endpoint or hardcoded bootstrap
- Nodes advertise themselves (IP:port, VRAM, capabilities)
- Coordinator discovers and connects to all nodes

**Outcome:** Automatic node discovery and connection.

### Phase 3: Layer Assignment
- Given N nodes with X VRAM each, distribute layers optimally
- Use topology solver to optimize for latency
- Support redundant layers for fault tolerance

**Outcome:** Automatic layer assignment across distributed nodes.

### Phase 4: Verification
- Redundant computation (compute on 2 nodes, compare)
- Test prompts with known answers
- Detect and handle malicious or faulty nodes

**Outcome:** Trust but verify — detect garbage output.

### Phase 5: Incentive Ledger
- Track contribution (uptime, VRAM, compute cycles)
- Allocate inference slots proportionally
- Simple ledger for trusted federation
- Research: distributed consensus for open network

**Outcome:** Fair resource allocation based on contribution.

---

## Technical Foundation

**What we're building:**
- Pipeline-parallel RPC → nodes can collaborate on inference
- Speculative decoding over WAN → hides latency
- Topology solver → optimizes node ordering
- Coordinator + direct-return → efficient draft/verify loop

**What's left for the decentralized layer:**
- Node discovery protocol
- Layer assignment algorithm
- Verification mechanism
- Incentive ledger

**Key insight:** Once Phase 1 is done, Phases 2-5 are mostly networking and distributed systems, not LLM-specific. The framework could be open-sourced for others to experiment with different incentive mechanisms.

---

## Cool Factor

This is essentially "Petals but with speculative decoding" — which means it could actually work over WAN instead of being bottlenecked by latency.

The research questions are hard but not impossible. The foundation we're building is real and valuable. The decentralized layer is a different beast, but it's a different scale of effort, not a different category.

**Next step:** Finish the pipeline-parallel RPC foundation, then start experimenting with the decentralized layer.

---

**Document version:** 1.0  
**Last updated:** 2026-06-20  
**Author:** Hermes Agent + icanplaytoo (brainstorm)
