# Fast-Platform O(1) Techniques for Satellite IR→RGB Super-Resolution + Colorization

**Research focus:** the genuinely O(1) (constant-time) or near-constant-time techniques for (A) data access / tile serving, (B) color / semantic mapping, and (C) model inference — assembled into one concrete fast-platform reference architecture.

**Method note:** Primary claims were checked via web research (June 2026). Items that could not be re-verified online are labelled `[from internal knowledge]` (knowledge cutoff Jan 2026). A recurring theme below: **very little is truly O(1)**; most "fast" geospatial tech is O(1) *per element* but O(log n) or O(n) in the size of the area/dataset. The report is explicit about which is which.

---

## TL;DR — The honest complexity picture

The single most important framing for this project:

- **Truly O(1):** array/LUT indexing, hash-map class→color, lat/lon→spatial-cell encoding (H3/S2/geohash/quadkey), chunk address computation in Zarr/TileDB, cache hits keyed by cell ID, and **per-tile** model inference at a fixed tile size (e.g. 512×512). These do not depend on how big the global archive is.
- **O(log n):** spatial range queries (R-tree / STAC search / GeoParquet predicate pushdown), B-tree/skiplist lookups, ANN graph traversal (HNSW), overview-level selection in a COG pyramid.
- **O(n):** scanning a scene, mosaicking many tiles, reading a non-cloud-optimized GeoTIFF, brute-force nearest-neighbor color search, processing an area that grows with zoom/extent.

The platform is "fast" precisely because it pushes the **hot path** (one user requesting one tile at one zoom) onto the O(1) primitives, and pre-computes or caches everything that would otherwise be O(n).

---

## A) DATA ACCESS / SERVING — O(1) tile reads

### A1. Cloud-Optimized GeoTIFF (COG) — *the cornerstone*
A COG is an ordinary GeoTIFF with three internal properties: **(1) internal tiling** (e.g. 256×256 or 512×512 blocks stored contiguously), **(2) overviews / pyramids** (pre-computed downsampled levels reducing resolution by 2×, 4×, 8×, 16×…), and **(3) byte layout designed for HTTP range requests**.

**Mechanics of the O(1)-ish read:** A client first issues one small range request (~16 KB) to read the header / IFDs (Image File Directories). The header tells it the exact byte offset and length of every internal tile at every overview level. To serve a given map tile the client then issues **one HTTP GET Range request** for just those bytes — it never downloads the whole scene. Because the COG's internal tiling can be aligned to the web-mercator tile grid, **one output tile ≈ one range request**.

**Honest complexity:** reading *one* tile is effectively **O(1)** (one header read amortized/cached + one range read of bounded size). Reading an *area* is O(number of tiles in that area). Picking the right overview is **O(log n)** in the zoom pyramid depth (tiny constant). Non-cloud-optimized GeoTIFFs are O(n) because you must scan/download to find data.

### A2. STAC + stac-geoparquet — spatial/temporal catalog lookup
**STAC (SpatioTemporal Asset Catalog)** is a JSON standard describing where assets (COGs) live, their footprint, datetime, and bands. A **STAC API** answers "what scenes cover this bbox/time?" Public deployments: **Microsoft Planetary Computer**, **Element84 Earth Search** (AWS Registry of Open Data — full Landsat Collection 2, Sentinel-1/2, Copernicus DEM, NAIP), and others.

**stac-geoparquet** stores STAC items as a columnar **GeoParquet** table for bulk/fast queries. With predicate pushdown + row-group statistics, a spatial-intersects filter prunes most of the file before reading. Both Planetary Computer and Earth Search publish geoparquet snapshots for this.

**Honest complexity:** STAC search is **O(log n)** at best (spatial index / partition pruning), not O(1). It is the *discovery* step, run rarely; the result (asset URLs) is cached so the hot path skips it.

### A3. Google Earth Engine (GEE) — server-side on-demand tiles
GEE uses **lazy evaluation**: a global expression isn't computed until a tile is requested, and then *only the tiles covering the requested region are computed*. Tile access via `ee.data.getTileUrl` / XYZ map tiles; static renders via `ee.Image.getThumbURL` (deprecated `size`, now `dimensions`; capped at 1e5×1e5 px).

**Pros:** zero data management, "publicly available supercomputer," works from thin clients. **Cons (critical for this project):** per-tile **memory limits**; complex per-pixel multi-band ops can exceed them; the **export pipeline is not designed for real-time/operational serving**; you do not control the runtime; throughput and latency are unpredictable under load. Best for prototyping/exploration, *not* as the production low-latency serving layer.

### A4. Open data buckets & tile servers
- **AWS Open Data / Element84 Earth Search**, **Microsoft Planetary Computer** — Landsat/Sentinel already as COGs in S3/Azure Blob, directly range-requestable.
- **TiTiler** (Development Seed) — FastAPI dynamic tile server: serves XYZ/WMTS tiles on the fly from COG / STAC / MosaicJSON / Zarr, with on-the-fly band math, algorithms, rescaling, and **colormap application**. Built on **rio-tiler**.
- **rio-tiler** — Rasterio plugin that does the actual "read just this tile from a COG" work; supports `morecantile` TileMatrixSets.
- **Terracotta** — lightweight tile server backed by an SQL metadata DB + (optimized) raster files; good when tiles/extents are known and you want a simple, fast, mostly-static service.
- **TileDB** — array database; raster as 2D dense arrays / 3D temporal stacks with built-in indexing and tiling (see A6).
- **MosaicJSON** — a JSON spec that maps a quadkey/tile → the list of COG asset URLs that cover it, enabling a **virtual mosaic** over thousands of scenes without a database; lookup of "which COGs for this quadkey" is a **hash/dict lookup ≈ O(1)** once the MosaicJSON is loaded.
- **XYZ / WMTS** — slippy-map addressing where `z/x/y` (or **quadkey**, the interleaved-bit encoding of x/y at zoom z) directly names a tile. Computing the quadkey for a tile is pure bit manipulation → **O(1)**; it is an ideal **cache key**.

### A5. Spatial indexing for O(1) addressing — H3 / S2 / geohash / quadkey
These map a `(lat, lon[, resolution])` to a compact hierarchical cell ID using **fixed-width bit/arithmetic operations**, independent of dataset size → **O(1)**.

- **Uber H3** — projects Earth onto an icosahedron, subdivides faces into **hexagons** (12 pentagons close the sphere); 16 resolutions (0–15), each cell ≈7 children; **64-bit integer** index. `latLngToCell` (formerly `geoToH3`) is **O(1)**. Hexagons give uniform, equidistant neighbors (good for blending/aggregation) but are *not* a clean hierarchy (children don't perfectly nest).
- **Google S2** — **spherical quadrilaterals** via a Hilbert space-filling curve into 64-bit cell IDs; clean parent/child hierarchy; excellent for range queries because nearby cells have nearby IDs.
- **Geohash** — interleaves lat/lon bits, Base32-encodes to a string; simple, clean hierarchy, but **rectangular cells with large size discrepancy** and edge-adjacency quirks.
- **Quadkey (Bing/slippy)** — base-4 string per zoom; trivially convertible to/from XYZ; the natural key for web-mercator tile caches.

**Use in this project:** compute `H3(lat,lon,res)` or `quadkey(z,x,y)` as the **cache key** for "has this tile already been super-resolved/colorized?" — an O(1) hash lookup decides cache-hit vs. compute. Benchmarks (e6data) show H3 vs quadkey differ mainly on join/aggregate workloads, not on the O(1) encode step.

> Complexity caveat: lat/lon→cell **encode** is O(1); a **k-ring / radius / range query** over cells is O(k) or O(log n), not O(1).

### A6. Data formats for fast ML I/O
- **Zarr** — cloud-native **chunked, compressed N-D arrays**; a chunk's storage key is computed by integer division of the index by chunk size → **address computation is O(1)**, and fetching one chunk is one object GET. Smaller chunks ⇒ faster random access. Heavily used by NASA/NOAA/ECMWF/Pangeo. With tuned PyTorch DataLoaders + Dask, ~15× throughput and GPU saturation reported.
- **TileDB** — dense/sparse array engine with native tiling + indexing; "data engine for ML"; good for SAR/temporal stacks.
- **WebDataset** `[from internal knowledge]` — POSIX **tar shards** streamed sequentially for maximum throughput training; great for *sequential* GPU feeding, **not** for O(1) random access to one tile (it trades random-access for streaming bandwidth).
- **Parquet / GeoParquet** — columnar; predicate + projection pushdown; the backbone of stac-geoparquet (A2).
- **FlatGeobuf** — streamable vector format with an optional **packed Hilbert R-tree** index → fast (O(log n)) spatial range reads over HTTP ranges; useful for vector overlays/masks, not raster pixels.
- **COG** (A1) — the raster equivalent for random tile access.

**Honest complexity:** chunked formats give **O(1) chunk-address math + O(1) single-chunk fetch**; reading a *region* is O(#chunks). They are random-access-friendly; WebDataset deliberately is not.

### A7. Caching — turning repeat reads into true O(1)
- **CDN / edge tile cache** (CloudFront, Cloudflare, Fastly) keyed by `z/x/y` or quadkey URL → O(1) hash lookup, served near the user; the dominant latency win for popular tiles.
- **Redis / Memcached** tile cache keyed by **H3/quadkey** → O(1) get/put for "already-rendered" tiles.
- **Memory-mapped arrays (`mmap`/`np.memmap`)** — OS page cache makes hot pixels an O(1) memory read.
- **LRU** (e.g. `functools.lru_cache`, or an LRU in front of rio-tiler) for COG headers/IFDs so the ~16 KB header read happens once.
- **Precomputed pyramids / pre-baked tiles** — if outputs are finite, render once to a COG/MBTiles pyramid and serve statically: every future read is O(1).

---

## B) O(1) COLOR / SEMANTIC MAPPING

### B1. Look-Up Tables (LUT) — the canonical O(1) color op
A **LUT** maps an input value to an output color by **direct array indexing** — genuinely **O(1)** per pixel, no branching.
- **1D LUT / palette / colormap** — `IR_value (0..255) → RGB`. This is exactly what TiTiler's `colormap` does on the fly. For class IDs (segmentation) it's `class_id → RGB`, an O(1) gather. Perfect for IR-intensity pseudocolor or class palettes.
- **3D LUT** — models a non-linear RGB→RGB transform by sparsely sampling a discretized 3D lattice and **trilinear interpolation** between the 8 surrounding lattice points → still **O(1) per pixel** (fixed 8-tap interpolation). This is the standard for film-grade color grading and is GPU-trivial.

### B2. Learned 3D LUTs — fast, image-adaptive color/colorization
- **"Learning Image-adaptive 3D LUTs for High-Performance Photo Enhancement in Real-time"** (Zeng et al.) — a tiny CNN (**<600K params**) predicts blend weights over a few **basis 3D LUTs**; processes **4K images in <2 ms on a Titan RTX (~500 FPS)** [verified from the paper abstract]. The expensive part is offloaded to the O(1) LUT lookup; the CNN runs once per image at thumbnail scale.
- **AdaInt (CVPR 2022)** — learns **non-uniform sampling intervals** (`AiLUT-Transform`) so the LUT spends resolution where the color transform is most non-linear; SOTA quality with negligible overhead; ideal for IR→RGB colorization where mapping is highly non-linear. Self-distilled variants exist (2025).

**Why this matters here:** the IR→RGB *colorization* refinement can be a **learned 3D LUT** applied after the SR network — adding near-zero latency while giving controllable, scene-adaptive color. This is the cleanest "O(1) color" lever in the whole pipeline.

### B3. Hash maps / perfect hashing for class→palette
For a fixed, known set of semantic classes, a **perfect hash** (or just a dense array if class IDs are contiguous) gives **guaranteed O(1)** `class → palette entry` with no collisions and a known table. For contiguous small ID ranges, the dense-array LUT (B1) is simplest and fastest.

### B4. Approximate Nearest Neighbor (ANN) for exemplar/reference color
For **example-based colorization** (find the nearest reference patch/feature and borrow its color), exact NN is O(n). ANN libraries give **sub-linear, amortized near-O(1)** retrieval:
- **FAISS** (Meta) — IVF partitioning + product quantization; probes only `nprobe` of `nlist` lists → ~`O(N/nlist × nprobe)`; GPU support.
- **ScaNN** (Google) — anisotropic vector quantization tuned for the speed/accuracy frontier.
- **HNSW** — navigable small-world **graph**; greedy descent gives roughly **O(log N)** query time, the usual default for low-latency vector search.

**Honest complexity:** ANN is **not O(1)** — it's O(log N) (HNSW) or O(N/partitions) (IVF), but *amortized near-constant* for fixed index size and is the practical choice for reference-color retrieval if you go that route. A learned 3D LUT (B2) avoids ANN entirely and is preferable when you don't need true exemplar transfer.

---

## C) MODEL INFERENCE SPEED

### C1. Export / compile / optimize runtimes
- **TensorRT** (NVIDIA) — best GPU latency; FP16/INT8/FP8 engines; layer/tensor fusion; the top choice for the serving GPU. Reported SR results: SPAN @ **95.6 FPS, 10.46 ms latency** for 2× via TensorRT on an A6000; depth-map nets hit **0.6–1.0 ms** at 256²–512² in FP16/INT8.
- **ONNX Runtime** — portable; CUDA / TensorRT / OpenVINO **execution providers**; SR models (EfRLFN, RLFN, SPAN) exceed real-time (≥30 FPS) in FP16.
- **OpenVINO** — best CPU/iGPU path (Intel) if no GPU.
- **torch.compile / TorchScript** — graph capture + kernel fusion without leaving PyTorch; easiest first win.

### C2. Numeric precision & model compression
- **FP16 / BF16** — ~2× throughput, near-lossless for SR; the default serving precision.
- **INT8 quantization** — further latency drop; use **QAT (quantization-aware training)** to recover accuracy (NVIDIA shows ~FP32 accuracy with QAT). Caveat: INT8 can occasionally be *slower* than FP16 on some layers/GPUs — benchmark, don't assume.
- **Pruning** (structured/channel) — fewer FLOPs.
- **Knowledge distillation** — train a small student to match a big teacher; key for shrinking SR/colorization nets.
- **Structural reparameterization** — train a multi-branch block, **collapse it to a single 3×3 conv at inference** for "free" speedups: **RepVGG** (origin), **ECBSR** (Edge-oriented Conv Block; 270p/540p→1080p **in real-time on a Snapdragon 865 mobile SoC**), **PlainUSR**, RepNet-VSR, and reparameterizable IR-SR nets (2025). Excellent fit for an efficient SR backbone here.

### C3. Architecture & diffusion choices
- **Efficient SR backbones** — RLFN/EfRLFN, SPAN, ECBSR, PlainUSR: lightweight CNNs that hit real-time at moderate scale.
- **One-step diffusion vs multi-step** — classic diffusion SR needs dozens–hundreds of steps (too slow). **Distilled one-step** models — **SinSR** (1-step distillation of ResShift; can blur), **OSEDiff** (distills SD via VSD), **InvSR**, **AdcSR**, **GenDR**, flow-trajectory-distillation (2025) — collapse this to **a single forward pass**. ResShift/UPSR show 15→4 step variants. For a tight latency budget, prefer a **one-step distilled** model or a plain efficient CNN over multi-step diffusion.

### C4. Tiled inference, batching, GPU streams
- **Patch-based tiled inference with overlap-blend** — split a large scene into fixed tiles (e.g. 512×512) with an overlap margin; run each tile; stitch with a **2-D weight window (linear ramp / raised-cosine / Gaussian / spline)** that down-weights tile edges and normalizes by the summed weights → **seamless, no visible seams**. Per-tile cost is **O(1)** (fixed size); whole-scene cost is O(#tiles) and **embarrassingly parallel**.
- **Batching** — group many tiles into one batched forward pass to saturate the GPU (raises tiles/sec).
- **CUDA streams / async H2D-D2H copy** — overlap data transfer with compute; pipeline decode → infer → encode.

### C5. Throughput / latency budget (per 512×512 tile, GPU, FP16/INT8)
Illustrative budget for the hot path of one tile (orders of magnitude, hardware-dependent):

| Stage | Typical latency | Complexity |
|---|---|---|
| Cache lookup (Redis/CDN, H3/quadkey key) | < 1 ms (hit ⇒ done) | **O(1)** |
| COG range read (1 tile, warm header) | 5–30 ms (network-bound) | **O(1)** per tile |
| Decode + preprocess (mmap/np) | 1–3 ms | O(pixels) = O(1) fixed tile |
| SR/colorization forward (TensorRT FP16/INT8, efficient/1-step) | 5–30 ms | **O(1)** per fixed tile |
| 3D-LUT color refinement | < 1 ms (trilinear) | **O(1)** per pixel |
| Encode tile → PNG/COG, serve | 1–5 ms | O(1) fixed tile |

Cache **miss** total ≈ **15–70 ms/tile**; cache **hit** ≈ **sub-millisecond to a few ms**. At ~20–30 ms/tile and batching, a single modern GPU sustains **tens of tiles/sec**, scaling linearly with GPUs/replicas.

---

## Reference Architecture (diagram-in-text)

```
                          ┌──────────────────────────────────────────────────────────┐
                          │  DISCOVERY (rare, O(log n)) — run once, cache the result   │
                          │  STAC API / stac-geoparquet  →  asset (COG) URLs           │
                          │  Sources: Planetary Computer · Earth Search · AWS Open Data│
                          │  (GEE optional, prototyping only — not the serving path)   │
                          └───────────────────────────┬──────────────────────────────┘
                                                      │ COG URLs + MosaicJSON (quadkey→COGs, O(1) dict)
                                                      ▼
  USER REQUEST  z/x/y ──►  quadkey / H3(lat,lon,res)  ──►  ┌──────────────────────────┐
   (one tile)            (O(1) bit-encode = CACHE KEY)     │  CDN edge cache  (O(1))  │──HIT──► return tile
                                                           │  Redis tile cache(O(1))  │           (sub-ms)
                                                           └────────────┬─────────────┘
                                                                        │ MISS
                                                                        ▼
                                       ┌────────────────────────────────────────────────┐
                                       │  O(1) TILE FETCH                                 │
                                       │  rio-tiler / TiTiler → 1 HTTP Range GET on COG   │
                                       │  (warm IFD header via LRU; Zarr/TileDB chunk     │
                                       │   read = O(1) chunk-address math)                │
                                       └───────────────────────┬────────────────────────┘
                                                               ▼
                                       ┌────────────────────────────────────────────────┐
                                       │  PREPROCESS (O(1)/fixed 512×512)                 │
                                       │  mmap decode · normalize · pad for overlap-blend │
                                       └───────────────────────┬────────────────────────┘
                                                               ▼
                                       ┌────────────────────────────────────────────────┐
                                       │  OPTIMIZED MODEL (O(1) per tile)                 │
                                       │  Efficient SR / 1-step distilled diffusion       │
                                       │  Reparam backbone (ECBSR/RepVGG) · TensorRT      │
                                       │  FP16/INT8 · batched · CUDA streams              │
                                       └───────────────────────┬────────────────────────┘
                                                               ▼
                                       ┌────────────────────────────────────────────────┐
                                       │  3D-LUT COLOR REFINEMENT (O(1) per pixel)        │
                                       │  Learned image-adaptive 3D LUT (AdaInt) +        │
                                       │  trilinear interp · IR→RGB / class→palette LUT   │
                                       └───────────────────────┬────────────────────────┘
                                                               ▼
                                       ┌────────────────────────────────────────────────┐
                                       │  OUTPUT + WRITE-BACK                             │
                                       │  encode → COG tile · TiTiler serves XYZ/WMTS     │
                                       │  write tile to Redis + CDN keyed by quadkey      │
                                       │  (next request for this tile = O(1) hit)         │
                                       └────────────────────────────────────────────────┘
```

**Overlap-blend note:** seams between adjacent SR tiles are removed by a raised-cosine/Gaussian 2-D weight window during stitch; per-tile work stays O(1), scene work is O(#tiles) and parallel.

---

## Master complexity & role table

| # | Technique / platform | Role | Hot-path complexity | Truly O(1)? |
|---|---|---|---|---|
| 1 | Cloud-Optimized GeoTIFF (COG) | Random tile read via HTTP Range | O(1) per tile (+O(log n) overview pick) | **Yes, per tile** |
| 2 | HTTP Range requests | Fetch only needed bytes | O(1) per range | **Yes** |
| 3 | COG overviews / pyramids | Pre-baked zoom levels | O(log n) select, O(1) read | Near (tiny const) |
| 4 | STAC API | Asset discovery by bbox/time | O(log n) | No |
| 5 | stac-geoparquet / GeoParquet | Bulk catalog query, pushdown | O(log n) | No |
| 6 | Planetary Computer / Earth Search / AWS Open Data | COG data sources | O(log n) discover | No |
| 7 | Google Earth Engine | Server-side on-demand tiles (proto) | unpredictable; not for serving | No |
| 8 | TiTiler / rio-tiler | Dynamic XYZ/WMTS tile server + colormap | O(1) per tile | **Yes, per tile** |
| 9 | Terracotta | Lightweight static-ish tile server | O(1)–O(log n) | Near |
| 10 | MosaicJSON | quadkey → COG-list virtual mosaic | O(1) dict lookup | **Yes** |
| 11 | XYZ / WMTS / quadkey addressing | Tile naming + cache key | O(1) encode | **Yes** |
| 12 | Uber H3 | lat/lon→hex cell, cache key, blend grid | O(1) encode (O(k) k-ring) | **Yes (encode)** |
| 13 | Google S2 | lat/lon→cell, range-friendly IDs | O(1) encode | **Yes (encode)** |
| 14 | Geohash | lat/lon→string cell | O(1) encode | **Yes (encode)** |
| 15 | Zarr | Chunked N-D array random access | O(1) chunk addr + 1 GET | **Yes (per chunk)** |
| 16 | TileDB | Array DB, tiled raster/temporal | O(1) cell/tile, O(log n) range | **Yes (per tile)** |
| 17 | WebDataset | High-throughput sequential training I/O | O(1) stream step (not random) | Streaming, not random |
| 18 | FlatGeobuf | Streamable vector + Hilbert R-tree | O(log n) range | No |
| 19 | CDN / Redis tile cache | O(1) serve of rendered tiles | O(1) hash get | **Yes** |
| 20 | mmap / np.memmap / LRU | OS-page / header caching | O(1) | **Yes** |
| 21 | 1D LUT / colormap / palette | IR→RGB & class→color | O(1) per pixel | **Yes** |
| 22 | 3D LUT (trilinear) | Non-linear color transform | O(1) per pixel (8-tap) | **Yes** |
| 23 | Learned image-adaptive 3D LUT (Zeng et al.) | Fast learned color grade (<2 ms @4K) | O(1) per pixel | **Yes** |
| 24 | AdaInt | Adaptive-interval learned 3D LUT | O(1) per pixel | **Yes** |
| 25 | Perfect hashing / dense array | class→palette | O(1) | **Yes** |
| 26 | FAISS (IVF+PQ) | Exemplar color retrieval | ~O(N/nlist·nprobe) | No (amortized fast) |
| 27 | ScaNN | Exemplar color retrieval | sub-linear | No |
| 28 | HNSW | Exemplar color retrieval | ~O(log N) | No |
| 29 | TensorRT | GPU inference engine (FP16/INT8/FP8) | O(1) per fixed tile | **Yes (per tile)** |
| 30 | ONNX Runtime (CUDA/TRT/OpenVINO EP) | Portable optimized inference | O(1) per fixed tile | **Yes (per tile)** |
| 31 | OpenVINO | CPU/iGPU inference | O(1) per fixed tile | **Yes (per tile)** |
| 32 | torch.compile / TorchScript | Graph fusion in PyTorch | O(1) per fixed tile | **Yes (per tile)** |
| 33 | FP16 / BF16 / INT8 quantization (+QAT) | ~2×+ throughput | constant-factor speedup | Speedup, not Big-O |
| 34 | Pruning / distillation | Smaller/faster model | constant-factor | Speedup |
| 35 | Structural reparam (RepVGG / ECBSR / PlainUSR) | Collapse to single 3×3 at inference | O(1) per fixed tile | **Yes (per tile)** |
| 36 | One-step distilled diffusion (SinSR/OSEDiff/InvSR) | Single-pass SR vs multi-step | O(1) per fixed tile | **Yes (per tile)** |
| 37 | Patch tiled inference + overlap-blend | Seamless large-scene SR | O(1)/tile, O(n) scene, parallel | Per-tile yes |
| 38 | Batching + CUDA streams | GPU saturation, pipelining | constant-factor | Speedup |

---

## Recommended fast-platform stack (concrete)

1. **Data sources:** Sentinel-2 / Landsat **COGs** from **Element84 Earth Search** + **Microsoft Planetary Computer** (AWS/Azure open data). Discovery via **STAC API** with a **stac-geoparquet** snapshot for bulk; GEE only for prototyping.
2. **Addressing & cache key:** **quadkey** (web-mercator serving) and/or **H3** (analytics/blend grid) — O(1) encode → key.
3. **Cache:** **CDN edge** (CloudFront/Cloudflare) in front of **Redis** tile cache; LRU on COG IFD headers; precompute pyramids for hot regions.
4. **Tile fetch:** **TiTiler + rio-tiler** issuing single HTTP Range reads on COGs; **MosaicJSON** for multi-scene virtual mosaics; **Zarr/TileDB** if you stage data as chunked arrays for ML.
5. **Model:** efficient SR backbone with **structural reparam (ECBSR-style)** *or* a **one-step distilled diffusion** SR, exported to **ONNX → TensorRT**, **FP16 (INT8 via QAT where it helps)**, **batched**, **CUDA-stream** pipelined, run as **patch-tiled inference with raised-cosine overlap-blend**.
6. **Color:** post-network **learned image-adaptive 3D LUT (AdaInt)** for IR→RGB colorization + a 1D **colormap/palette LUT** for class/intensity overlays — all O(1) per pixel. Use **FAISS/HNSW** only if you specifically need exemplar/reference color transfer.
7. **Output/serving:** encode to **COG tiles**, serve **XYZ/WMTS via TiTiler**, write results back to Redis+CDN so repeat tiles are **O(1) cache hits**.

**Bottom line on "O(1):"** the platform is fast because the per-request hot path is dominated by genuinely O(1) primitives — **cache lookup, COG range read, fixed-size tile inference, and LUT color** — while the inherently O(log n)/O(n) work (catalog search, mosaicking, pyramid build) is done **rarely and/or pre-computed**, then cached behind an O(1) key.

---

## Sources

- The Complete Guide to Cloud Optimized GeoTIFF (COG): https://atlas.co/blog/the-complete-guide-to-cloud-optimized-geotiff-cog/
- COG specification (cogeotiff): https://github.com/cogeotiff/cog-spec/blob/master/spec.md
- OGC Cloud Optimized GeoTIFF Standard: https://docs.ogc.org/is/21-026/21-026.html
- Cloud-Optimized Geospatial Formats Guide — COG details: https://guide.cloudnativegeo.org/cloud-optimized-geotiffs/cogs-details.html
- Dynamic map tiling with COGs (Kyle Barron): https://kylebarron.dev/blog/cog-mosaic/overview/
- stac-geoparquet (Gadomski): https://www.gadom.ski/presentations/2025-11-04-stac-geoparquet.html
- Planetary Computer — bulk STAC item queries with GeoParquet: https://planetarycomputer.microsoft.com/docs/quickstarts/stac-geoparquet/
- Element84 Earth Search: https://element84.com/earth-search/ and examples: https://element84.com/earth-search/examples
- How Microsoft's Planetary Computer uses STAC (Element84): https://element84.com/geospatial/how-microsofts-planetary-computer-uses-stac/
- TiTiler (Development Seed): https://developmentseed.org/titiler/ and repo: https://github.com/developmentseed/titiler
- rio-tiler: https://github.com/cogeotiff/rio-tiler
- GEE client vs server: https://developers.google.com/earth-engine/guides/client_server
- GEE getThumbURL: https://developers.google.com/earth-engine/apidocs/ee-image-getthumburl
- GEE getTileUrl: https://developers.google.com/earth-engine/apidocs/ee-data-gettileurl
- Google Earth Engine power & limitations (Off-Nadir Delta): https://offnadir-delta.com/blog/google-earth-engine-satellite-analysis
- Geospatial indexing explained — Geohash/S2/H3 (Ben Feifke): https://benfeifke.com/posts/geospatial-indexing-explained/
- H3 docs: https://h3geo.org/docs/ and repo: https://github.com/uber/h3 and Uber blog: https://www.uber.com/us/en/blog/h3/
- H3 vs Quadkey performance (e6data): https://www.e6data.com/blog/geospatial-analytics-performance-bottleneck-h3-vs-quadkey-for-spatial-indexing
- What is Zarr (Earthmover): https://www.earthmover.io/blog/what-is-zarr/ and Zarr (Wikipedia): https://en.wikipedia.org/wiki/Zarr_(data_format)
- Optimizing Cloud-to-GPU throughput for EO data (arXiv): https://arxiv.org/pdf/2506.06235
- TileDB as the data engine for ML: https://www.tiledb.com/blog/tiledb-as-the-data-engine-for-machine-learning
- FAISS/HNSW/ScaNN roles (Milvus): https://milvus.io/ai-quick-reference/what-is-the-role-of-faiss-hnsw-and-scann-in-ai-databases
- HNSW (Pinecone): https://www.pinecone.io/learn/series/faiss/hnsw/
- Learning Image-adaptive 3D LUTs (arXiv 2009.14468): https://arxiv.org/abs/2009.14468
- AdaInt (arXiv 2204.13983): https://arxiv.org/abs/2204.13983
- Self-distilled adaptive-interval 3D LUTs (2025, ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S0031320325012622
- Exploring Real-Time Super-Resolution benchmarking (arXiv): https://arxiv.org/pdf/2602.11339
- Optimizing/deploying transformer INT8 with ONNX Runtime-TensorRT (Microsoft): https://opensource.microsoft.com/blog/2022/05/02/optimizing-and-deploying-transformer-int8-inference-with-onnx-runtime-tensorrt-on-nvidia-gpus/
- INT8 via QAT with TensorRT (NVIDIA): https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/
- SinSR — single-step diffusion SR: https://research.polyu.edu.hk/en/publications/sinsr-diffusion-based-image-super-resolution-in-a-single-step/
- One-step residual-shifting diffusion via distillation (arXiv): https://arxiv.org/pdf/2503.13358
- ECBSR — Edge-oriented Convolution Block for Real-time SR: https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf
- PlainUSR — faster ConvNet for efficient SR (ACCV 2024): https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf
- Reparameterizable large-kernel attention for IR-image SR (Sci. Reports 2025): https://www.nature.com/articles/s41598-025-24193-3
- Tiled inference / smooth blending (GeoAI): https://opengeoai.org/inference/ and https://opengeoai.org/examples/smooth_inference/
- SAHI tiled inference (Ultralytics): https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/sahi-tiled-inference.md
