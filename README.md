
# Descriptron × GBIF Annotator — Web Morphology Annotator

A **single-page web application (SPA)** implemented as a **single-file HTML/JavaScript app** for morphological annotation of specimen images (e.g., from GBIF), with an optional backend for image proxying/uploads and (optionally) SAM2-assisted segmentation support.

<img width="1004" height="861" alt="Screenshot from 2026-03-02 15-12-07" src="https://github.com/alexrvandam/Descriptron-GBIF_Annotator/blob/main/Screenshot%20from%202026-03-02%2015-12-07.png" />

## Live deployment

This GitHub repository contains the source files (frontend + optional backend/Docker setup) to reproduce the project.

A fully working hosted version is available on a Hetzner server at:
[https://descriptrongbifannotator.org/
](https://descriptrongbifannotator.org/) 

if for some reason you have fortinet or another firewall you can also try the server directly via :[http://46.225.84.116:8100/](http://46.225.84.116:8100/)


## What's new in v1.0.6 (30 July 2026)

This release addresses issues raised during peer review of the accompanying manuscript:

- **Self-hosted AI runtime (fixes `ONNX decoder failed`).** The ONNX Runtime Web library is no
  longer loaded from a third-party CDN — all three runtime files now ship in `static/ort/` and are
  served from the same origin as the app. Some antivirus/firewall products block CDN requests for
  WebAssembly, which silently broke in-browser segmentation. Loading is now same-origin first, with
  automatic fallback to **server-side** segmentation if the browser runtime cannot start at all.
- **Interactive guided tour.** First-time visitors get a step-through walkthrough of the whole
  workflow (skippable), re-openable any time from the **❓ Tour** button in the header.
- **Sign-in fix** for host/origin mismatches between the apex and `www.` domains.
- **TLS note for self-hosters:** certbot renewal does *not* cause a long-running nginx to load the
  new certificate — schedule a periodic `nginx -s reload` independently of renewal, or browsers will
  keep seeing the old (eventually expired) certificate while it looks valid on disk.

## What’s in this repository

### Frontend (single-file web app) 
- `descriptron-gbif-annotator.html` (you may rename to `index.html`)

The frontend runs in the browser and provides:
- Loading images by URL (including GBIF media URLs)
- UI for annotation workflows (template/view driven)
- Interactive drawing/annotation tools (boxes/points/masks depending on configuration)
- Export of annotations to research-friendly formats (project-dependent)
- Optional publishing / workflow integrations if enabled in the UI

### Backend (optional, but recommended for full functionality)
- `app.py` — API server used for:
  - `GET /health` (service status)
  - `GET /proxy-image?url=...` (avoid CORS issues when fetching remote images)
  - `POST /upload-image` (upload local images)
  - `GET /onnx/decoder.onnx` (serve ONNX decoder artifact, if used)
  - Optional SAM-related endpoints like `POST /encode` (if enabled)
- `Dockerfile`
- `docker-compose.yml`
- `static/` (static assets served by the backend)
- `static/ort/` (self-hosted ONNX Runtime Web: `ort.all.min.js`,
  `ort-wasm-simd-threaded.jsep.mjs`, `ort-wasm-simd-threaded.jsep.wasm`).
  **If you upgrade onnxruntime-web, replace all three files *and* the version in the HTML
  `<script>` fallback URL together — the loader and the WASM binary must match.**

> Notes:
> - Runtime/telemetry files (e.g., `usage.db`) and server-specific infrastructure are intentionally excluded from version control.

## Quickstart (UI only)


[!Checkout this video on YouTube about Descriptron-GBIF Annotator (https://github.com/alexrvandam/Descriptron-GBIF_Annotator/blob/main/video-thumbnail_my_crop.jpg) ](https://youtu.be/D9C9fcGsaA0)

if you use the website or github or find it useful please cite it,

Van Dam, A.R. & Hita Garcia, F. Descriptron-GBIF Annotator (Version 1.0.6) [Computer software]. https://github.com/alexrvandam/Descriptron-GBIF_Annotator. doi: https://doi.org/10.5281/zenodo.18888577 (all versions)

@software{VanDam_DescriptronGBIFAnnotator_2026,
  author  = {Van Dam, Alex},
  title   = {Descriptron-GBIF Annotator},
  version = {1.0.6},
  doi     = {10.5281/zenodo.18888577},
  url     = {https://github.com/alexrvandam/Descriptron-GBIF_Annotator},
  note    = {Archived on Zenodo: \url{https://doi.org/10.5281/zenodo.18888577}}
}


