// ═══════════════════════════════════════════════════════════════════════
// SAM2.1 ONNX Split Architecture — Client-Side Decoder
// ═══════════════════════════════════════════════════════════════════════
//
// Flow:
//   1. Browser loads ONNX decoder model once (~10MB, cached)
//   2. User selects specimen → server encodes image (~15-30s CPU)
//   3. Server returns embedding tensors as binary blob
//   4. Each point/bbox prompt → decoder runs IN BROWSER (~10ms)
//   5. Masks appear instantly, no network round-trip per click
//
// Dependencies:
//   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.all.min.js"></script>
//
// ═══════════════════════════════════════════════════════════════════════

const SAM2Client = {
  // ─── State ───────────────────────────────────────────────────────
  apiBaseUrl: '',
  decoderSession: null,    // ONNX InferenceSession
  decoderLoading: false,
  decoderReady: false,

  // Current image embedding (from server)
  currentEmbedding: null,  // { image_embed, high_res_feats_0, high_res_feats_1 }
  currentOrigSize: null,   // { w, h }
  currentInputSize: null,  // { w, h } (after resize to 1024)
  encodingInProgress: false,

  connected: false,

  // ─── Initialize ──────────────────────────────────────────────────
  init(apiUrl) {
    if (apiUrl) {
      this.apiBaseUrl = apiUrl.replace(/\/+$/, '');
    } else if (window.location.protocol !== 'file:') {
      this.apiBaseUrl = window.location.origin + '/gbif-api';
    }
  },

  setUrl(url) {
    this.apiBaseUrl = url.replace(/\/+$/, '');
  },

  // ─── Health Check ────────────────────────────────────────────────
  async checkHealth() {
    if (!this.apiBaseUrl) return false;
    try {
      const r = await fetch(`${this.apiBaseUrl}/health`, { signal: AbortSignal.timeout(5000) });
      const d = await r.json();
      this.connected = d.status === 'ok' && d.encoder_loaded;
      return {
        connected: this.connected,
        device: d.device,
        onnxAvailable: d.onnx_decoder_available,
        cache: d.cache,
      };
    } catch (e) {
      this.connected = false;
      return { connected: false, error: e.message };
    }
  },

  // ─── CORS Image Proxy ────────────────────────────────────────────
  getProxyUrl(imageUrl) {
    if (!this.apiBaseUrl) return imageUrl;
    return `${this.apiBaseUrl}/proxy-image?url=${encodeURIComponent(imageUrl)}`;
  },

  // ─── Load ONNX Decoder (once) ────────────────────────────────────
  async loadDecoder(onProgress) {
    if (this.decoderReady) return true;
    if (this.decoderLoading) return false;

    this.decoderLoading = true;
    if (onProgress) onProgress('loading', 'Downloading ONNX decoder (~10MB)...');

    try {
      const decoderUrl = `${this.apiBaseUrl}/onnx/decoder.onnx`;

      // Configure ONNX Runtime
      const options = {
        executionProviders: ['wasm'],  // WebAssembly backend (universal)
        graphOptimizationLevel: 'all',
      };

      // Try WebGL first for GPU acceleration, fall back to WASM
      try {
        const testSession = await ort.InferenceSession.create(decoderUrl, {
          executionProviders: ['webgl'],
        });
        testSession.release();
        options.executionProviders = ['webgl', 'wasm'];
        if (onProgress) onProgress('loading', 'Using WebGL acceleration');
      } catch (e) {
        if (onProgress) onProgress('loading', 'Using WebAssembly (CPU)');
      }

      this.decoderSession = await ort.InferenceSession.create(decoderUrl, options);
      this.decoderReady = true;
      this.decoderLoading = false;

      if (onProgress) onProgress('ready', 'ONNX decoder loaded');
      console.log('SAM2 ONNX decoder loaded. Inputs:', this.decoderSession.inputNames,
                   'Outputs:', this.decoderSession.outputNames);
      return true;

    } catch (e) {
      this.decoderLoading = false;
      console.error('Failed to load ONNX decoder:', e);
      if (onProgress) onProgress('error', `Decoder load failed: ${e.message}`);
      return false;
    }
  },

  // ─── Encode Image (server-side) ──────────────────────────────────
  async encodeImage(imageUrl, onProgress) {
    if (!this.connected) throw new Error('Not connected to SAM2 API');

    this.encodingInProgress = true;
    this.currentEmbedding = null;
    if (onProgress) onProgress('encoding', 'Encoding image (may take 15-30s)...');

    try {
      const resp = await fetch(`${this.apiBaseUrl}/encode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: imageUrl }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Encoding failed');
      }

      // Read binary embedding
      const buffer = await resp.arrayBuffer();
      const cached = resp.headers.get('X-Cached') === 'true';
      const origW = parseInt(resp.headers.get('X-Orig-Width') || '0');
      const origH = parseInt(resp.headers.get('X-Orig-Height') || '0');

      // Unpack the binary format
      const embedding = this.unpackEmbeddings(buffer);

      this.currentEmbedding = embedding.tensors;
      this.currentOrigSize = { w: embedding.origW || origW, h: embedding.origH || origH };
      this.currentInputSize = { w: embedding.inputW, h: embedding.inputH };

      this.encodingInProgress = false;

      const msg = cached ? 'Loaded from cache' : `Encoded (${(buffer.byteLength/1024).toFixed(0)}KB)`;
      if (onProgress) onProgress('ready', msg);

      // Also load decoder if not loaded yet
      if (!this.decoderReady) {
        this.loadDecoder(onProgress);
      }

      return { cached, origSize: this.currentOrigSize, inputSize: this.currentInputSize };

    } catch (e) {
      this.encodingInProgress = false;
      if (onProgress) onProgress('error', e.message);
      throw e;
    }
  },

  // ─── Unpack Binary Embeddings ────────────────────────────────────
  unpackEmbeddings(buffer) {
    const view = new DataView(buffer);
    let offset = 0;

    // Header (32 bytes)
    const magic = String.fromCharCode(
      view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
    );
    if (magic !== 'SAM2') throw new Error('Invalid embedding format');
    offset = 4;

    const version = view.getUint32(offset, true); offset += 4;
    const origH = view.getUint32(offset, true); offset += 4;
    const origW = view.getUint32(offset, true); offset += 4;
    const inputH = view.getUint32(offset, true); offset += 4;
    const inputW = view.getUint32(offset, true); offset += 4;
    const numTensors = view.getUint32(offset, true); offset += 4;
    offset += 4; // reserved

    const tensors = {};
    for (let t = 0; t < numTensors; t++) {
      // Name
      const nameLen = view.getUint32(offset, true); offset += 4;
      const nameBytes = new Uint8Array(buffer, offset, nameLen);
      const name = new TextDecoder().decode(nameBytes); offset += nameLen;

      // Shape
      const ndim = view.getUint32(offset, true); offset += 4;
      const shape = [];
      for (let d = 0; d < ndim; d++) {
        shape.push(view.getUint32(offset, true)); offset += 4;
      }

      // Dtype (0=float32, 1=float16)
      const dtype = view.getUint32(offset, true); offset += 4;

      // Data
      const dataLen = view.getUint32(offset, true); offset += 4;
      const rawData = new Uint8Array(buffer, offset, dataLen); offset += dataLen;

      // Convert float16 to float32 for ONNX Runtime
      let float32Data;
      if (dtype === 1) {
        // float16 -> float32
        const f16 = new Uint16Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 2);
        float32Data = new Float32Array(f16.length);
        for (let i = 0; i < f16.length; i++) {
          float32Data[i] = float16ToFloat32(f16[i]);
        }
      } else {
        float32Data = new Float32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
      }

      tensors[name] = { data: float32Data, shape: shape };
    }

    return { tensors, origH, origW, inputH, inputW, version };
  },

  // ─── Run Decoder (client-side, ~10ms) ────────────────────────────
  async predict(prompts, usePreviousMask = null) {
    if (!this.decoderReady) throw new Error('ONNX decoder not loaded');
    if (!this.currentEmbedding) throw new Error('No image encoded');

    const embed = this.currentEmbedding;

    // Build prompt tensors
    // SAM2 point labels: 1=positive, 0=negative, 2=bbox-TL, 3=bbox-BR, -1=padding
    const allCoords = [];
    const allLabels = [];

    // Add point prompts
    if (prompts.points) {
      prompts.points.forEach(p => {
        // Scale from original image coords to input coords (1024-based)
        const sx = (this.currentInputSize.w / this.currentOrigSize.w) * p.x;
        const sy = (this.currentInputSize.h / this.currentOrigSize.h) * p.y;
        allCoords.push(sx, sy);
        allLabels.push(p.positive !== false ? 1 : 0);
      });
    }

    // Add bbox prompts (encoded as two points: TL and BR)
    if (prompts.bboxes) {
      prompts.bboxes.forEach(b => {
        const scaleX = this.currentInputSize.w / this.currentOrigSize.w;
        const scaleY = this.currentInputSize.h / this.currentOrigSize.h;
        allCoords.push(b.x * scaleX, b.y * scaleY);
        allLabels.push(2); // top-left
        allCoords.push((b.x + b.w) * scaleX, (b.y + b.h) * scaleY);
        allLabels.push(3); // bottom-right
      });
    }

    if (allCoords.length === 0) throw new Error('No prompts provided');

    const numPoints = allLabels.length;
    const pointCoords = new Float32Array(allCoords);   // [N*2] flat
    const pointLabels = new Float32Array(allLabels);    // [N]

    // Reshape for ONNX: [1, N, 2] and [1, N]
    const coordsTensor = new ort.Tensor('float32', pointCoords, [1, numPoints, 2]);
    const labelsTensor = new ort.Tensor('float32', pointLabels, [1, numPoints]);

    // Image embedding tensors
    const imageEmbed = new ort.Tensor('float32',
      embed.image_embed.data, embed.image_embed.shape);
    const hiRes0 = new ort.Tensor('float32',
      embed.high_res_feats_0.data, embed.high_res_feats_0.shape);
    const hiRes1 = new ort.Tensor('float32',
      embed.high_res_feats_1.data, embed.high_res_feats_1.shape);

    // Mask input (previous mask or zeros)
    let maskInput, hasMaskInput;
    if (usePreviousMask) {
      maskInput = new ort.Tensor('float32', usePreviousMask, [1, 1, 256, 256]);
      hasMaskInput = new ort.Tensor('float32', new Float32Array([1]), [1]);
    } else {
      maskInput = new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]);
      hasMaskInput = new ort.Tensor('float32', new Float32Array([0]), [1]);
    }

    // Run decoder
    const t0 = performance.now();
    const feeds = {
      image_embed: imageEmbed,
      high_res_feats_0: hiRes0,
      high_res_feats_1: hiRes1,
      point_coords: coordsTensor,
      point_labels: labelsTensor,
      mask_input: maskInput,
      has_mask_input: hasMaskInput,
    };

    const results = await this.decoderSession.run(feeds);
    const t1 = performance.now();

    // Parse outputs
    const masks = results.masks;          // [1, M, 256, 256]
    const iouScores = results.iou_predictions;  // [1, M]
    const lowResMasks = results.low_res_masks;

    const numMasks = masks.dims[1];
    const maskH = masks.dims[2];
    const maskW = masks.dims[3];

    const decoded = [];
    for (let i = 0; i < numMasks; i++) {
      const score = iouScores.data[i];
      // Extract mask [maskH, maskW]
      const start = i * maskH * maskW;
      const maskData = masks.data.slice(start, start + maskH * maskW);

      // Threshold at 0 (logits)
      const binaryMask = new Uint8Array(maskH * maskW);
      for (let j = 0; j < maskData.length; j++) {
        binaryMask[j] = maskData[j] > 0 ? 1 : 0;
      }

      // Convert to polygons for canvas rendering
      const polygons = maskToPolygons(binaryMask, maskW, maskH);

      // Low-res mask for iterative refinement
      const lrStart = i * 256 * 256;
      const lowRes = lowResMasks ? lowResMasks.data.slice(lrStart, lrStart + 256 * 256) : null;

      decoded.push({
        index: i,
        score: score,
        maskData: binaryMask,
        maskW, maskH,
        polygons,
        lowResMask: lowRes,
      });
    }

    // Sort by score
    decoded.sort((a, b) => b.score - a.score);

    console.log(`SAM2 predict: ${numMasks} masks in ${(t1-t0).toFixed(1)}ms, best=${decoded[0]?.score.toFixed(3)}`);

    return {
      masks: decoded,
      timing_ms: t1 - t0,
    };
  },

  // ─── Render mask on canvas ───────────────────────────────────────
  drawMask(ctx, mask, color, imgDims, opacity = 0.35) {
    // mask.maskData is 256x256, need to scale to image coordinates
    const { scale, ox, oy } = imgDims;
    const origW = this.currentOrigSize.w;
    const origH = this.currentOrigSize.h;

    // Create a temporary canvas for the mask
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = mask.maskW;
    tmpCanvas.height = mask.maskH;
    const tmpCtx = tmpCanvas.getContext('2d');
    const imgData = tmpCtx.createImageData(mask.maskW, mask.maskH);

    // Parse color
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    for (let i = 0; i < mask.maskData.length; i++) {
      const idx = i * 4;
      if (mask.maskData[i]) {
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = Math.round(opacity * 255);
      }
    }
    tmpCtx.putImageData(imgData, 0, 0);

    // Draw scaled to the image area on the main canvas
    const drawW = origW * scale;
    const drawH = origH * scale;
    ctx.drawImage(tmpCanvas, ox, oy, drawW, drawH);

    // Draw outline using polygons
    if (mask.polygons && mask.polygons.length > 0) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.9;

      // Scale polygon coords from mask space to canvas space
      const polyScaleX = (origW * scale) / mask.maskW;
      const polyScaleY = (origH * scale) / mask.maskH;

      mask.polygons.forEach(poly => {
        if (poly.length < 6) return;
        ctx.beginPath();
        ctx.moveTo(poly[0] * polyScaleX + ox, poly[1] * polyScaleY + oy);
        for (let i = 2; i < poly.length; i += 2) {
          ctx.lineTo(poly[i] * polyScaleX + ox, poly[i + 1] * polyScaleY + oy);
        }
        ctx.closePath();
        ctx.stroke();
      });
      ctx.restore();
    }
  },
};


// ─── Float16 → Float32 conversion ─────────────────────────────────
function float16ToFloat32(h) {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;

  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0 : 0;
    // Subnormal
    let e = -14;
    let m = mantissa;
    while (!(m & 0x400)) { m <<= 1; e--; }
    m &= 0x3ff;
    const f = (sign ? -1 : 1) * Math.pow(2, e) * (1 + m / 1024);
    return f;
  }
  if (exponent === 31) {
    return mantissa ? NaN : (sign ? -Infinity : Infinity);
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}


// ─── Mask to polygons (client-side contour tracing) ─────────────────
function maskToPolygons(binaryMask, width, height) {
  // Simple marching squares contour extraction
  const polygons = [];
  const visited = new Uint8Array(width * height);

  function getPixel(x, y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    return binaryMask[y * width + x];
  }

  // Find contour starting points (boundary pixels)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!getPixel(x, y) || visited[y * width + x]) continue;

      // Check if this is a boundary pixel
      const isBoundary = !getPixel(x - 1, y) || !getPixel(x + 1, y) ||
                         !getPixel(x, y - 1) || !getPixel(x, y + 1);
      if (!isBoundary) continue;

      // Trace contour using simple boundary following
      const contour = [];
      let cx = x, cy = y;
      let dir = 0; // 0=right, 1=down, 2=left, 3=up
      const startX = x, startY = y;
      let steps = 0;
      const maxSteps = width * height;

      do {
        contour.push(cx, cy);
        visited[cy * width + cx] = 1;

        // Try turning left, straight, right, back
        let found = false;
        for (let turn = -1; turn <= 2; turn++) {
          const newDir = (dir + turn + 4) % 4;
          const dx = [1, 0, -1, 0][newDir];
          const dy = [0, 1, 0, -1][newDir];
          const nx = cx + dx, ny = cy + dy;
          if (getPixel(nx, ny)) {
            const isEdge = !getPixel(nx - 1, ny) || !getPixel(nx + 1, ny) ||
                          !getPixel(nx, ny - 1) || !getPixel(nx, ny + 1);
            if (isEdge) {
              cx = nx; cy = ny; dir = newDir; found = true; break;
            }
          }
        }
        if (!found) break;
        steps++;
      } while ((cx !== startX || cy !== startY) && steps < maxSteps);

      // Only keep contours with enough points
      if (contour.length >= 6) {
        // Simplify (take every Nth point for large contours)
        if (contour.length > 200) {
          const step = Math.ceil(contour.length / 200) * 2; // *2 because flat [x,y,x,y...]
          const simplified = [];
          for (let i = 0; i < contour.length; i += step) {
            simplified.push(contour[i], contour[i + 1] || contour[contour.length - 1]);
          }
          polygons.push(simplified);
        } else {
          polygons.push(contour);
        }
      }
    }
  }

  return polygons;
}


// ═══════════════════════════════════════════════════════════════════════
// INTEGRATION INTO YOUR ANNOTATOR HTML
// ═══════════════════════════════════════════════════════════════════════
//
// In your HTML, add this script AFTER onnxruntime-web:
//
//   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.all.min.js"></script>
//   <script src="sam2-client.js"></script>
//
// Then in your init:
//
//   SAM2Client.init();  // auto-detect API URL
//   const health = await SAM2Client.checkHealth();
//   if (health.connected) {
//     await SAM2Client.loadDecoder();  // downloads ~10MB ONNX, cached by browser
//   }
//
// When image loads:
//
//   await SAM2Client.encodeImage(originalGbifUrl);  // ~15-30s, one-time
//
// On each click/bbox (runs locally, ~10ms):
//
//   const result = await SAM2Client.predict({
//     points: [{ x: 512, y: 384, positive: true }],
//     bboxes: [],
//   });
//   // result.masks[0] is the best mask
//   // Draw it:
//   SAM2Client.drawMask(overlayCtx, result.masks[0], '#ef4444', imgDims);
//
// For CORS-free image loading:
//
//   const proxyUrl = SAM2Client.getProxyUrl(gbifImageUrl);
//   img.src = proxyUrl;  // loads through server proxy, no CORS issues
//
// ═══════════════════════════════════════════════════════════════════════
