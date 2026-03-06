
# Descriptron × GBIF Annotator — Web Morphology Annotator

A **single-page web application (SPA)** implemented as a **single-file HTML/JavaScript app** for morphological annotation of specimen images (e.g., from GBIF), with an optional backend for image proxying/uploads and (optionally) SAM2-assisted segmentation support.

<img width="1004" height="861" alt="Screenshot from 2026-03-02 15-12-07" src="https://github.com/user-attachments/assets/b030956e-3560-4e6f-9565-d22103717afe" />

## Live deployment

This GitHub repository contains the source files (frontend + optional backend/Docker setup) to reproduce the project.

A fully working hosted version is available on a Hetzner server at:
[https://descriptrongbifannotator.org/
](https://descriptrongbifannotator.org/) 

if for some reason you have fortinet or another firewall you can also try the server directly via :[http://46.225.84.116:8100/](http://46.225.84.116:8100/)


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

> Notes:
> - Runtime/telemetry files (e.g., `usage.db`) and server-specific infrastructure are intentionally excluded from version control.

## Quickstart (UI only)

You can run the frontend without Docker, but some browser features (fetch/CORS) may be limited if you open the file directly.

### Option 1: Serve locally with Python
```bash
python3 -m http.server 8000

```
if you use the website or github or find it useful please cite it,

Van Dam, A. Descriptron-GBIF Annotator (Version 1.0.1) [Computer software]. https://github.com/alexrvandam/Descriptron-GBIF_Annotator. doi: https://doi.org/10.5281/zenodo.18888578

@software{VanDam_DescriptronGBIFAnnotator_2026,
  author  = {Van Dam, Alex},
  title   = {Descriptron-GBIF Annotator},
  version = {1.0.1},
  doi     = {10.5281/zenodo.18888578},
  url     = {https://github.com/alexrvandam/Descriptron-GBIF_Annotator},
  note    = {Archived on Zenodo: \url{https://doi.org/10.5281/zenodo.18888578}}
}


