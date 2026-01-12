const fileInput = document.getElementById("file-input");
const processBtn = document.getElementById("process");
const statusEl = document.getElementById("status");
const radiusEl = document.getElementById("radius");
const radiusValue = document.getElementById("radius-value");
const blendEl = document.getElementById("blend");
const blendValue = document.getElementById("blend-value");
const origCanvas = document.getElementById("orig");
const filteredCanvas = document.getElementById("filtered");

const origCtx = origCanvas.getContext("2d");
const filteredCtx = filteredCanvas.getContext("2d");

let adapter;
let device;
let imageBitmap = null;
let computePipeline = null;
let uniformBuffer = null;

const shaderCode = /* wgsl */ `
struct Params {
  radius : u32,
  _pad0  : u32,
  blend  : f32,
  _pad1  : f32,
};

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(srcTex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let origin = textureLoad(srcTex, vec2<i32>(gid.xy), 0);

  let r = i32(params.radius);
  var sum = vec3<f32>(0.0);
  var count = 0.0;

  var y = -r;
  loop {
    if (y > r) { break; }
    var x = -r;
    loop {
      if (x > r) { break; }
      let coord = clamp(vec2<i32>(gid.xy) + vec2<i32>(x, y), vec2<i32>(0, 0), vec2<i32>(dims) - vec2<i32>(1, 1));
      let sample = textureLoad(srcTex, coord, 0);
      sum += sample.rgb;
      count += 1.0;
      x += 1;
    }
    y += 1;
  }

  let blurred = sum / count;
  let mixed = mix(origin.rgb, blurred, params.blend);
  textureStore(dstTex, vec2<i32>(gid.xy), vec4<f32>(mixed, origin.a));
}
`;

init();

async function init() {
  if (!("gpu" in navigator)) {
    setStatus("WebGPU not available. Use Chrome/Edge 121+ with WebGPU enabled.", true);
    return;
  }

  adapter = await navigator.gpu.requestAdapter();
  device = await adapter?.requestDevice();

  if (!device) {
    setStatus("Unable to init WebGPU device.", true);
    return;
  }

  computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: shaderCode }),
      entryPoint: "main",
    },
  });

  uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  setStatus("WebGPU ready. Load an image to begin analysis.");
  processBtn.disabled = true;
}

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    imageBitmap = await createImageBitmap(await fetch(url).then((r) => r.blob()));
  } catch (err) {
    console.error(err);
    setStatus("Could not read image.", true);
    return;
  } finally {
    URL.revokeObjectURL(url);
  }

  resizeCanvases(imageBitmap.width, imageBitmap.height);
  origCtx.clearRect(0, 0, origCanvas.width, origCanvas.height);
  origCtx.drawImage(imageBitmap, 0, 0, origCanvas.width, origCanvas.height);
  processBtn.disabled = !device;
  setStatus("Image loaded. Adjust radius/blend then run the robustness test.");
});

radiusEl.addEventListener("input", () => {
  radiusValue.textContent = radiusEl.value;
});

blendEl.addEventListener("input", () => {
  blendValue.textContent = blendEl.value;
});

processBtn.addEventListener("click", () => {
  if (!imageBitmap) return;
  runDenoise().catch((err) => {
    console.error(err);
    setStatus("WebGPU processing failed.", true);
  });
});

function resizeCanvases(width, height) {
  [origCanvas, filteredCanvas].forEach((canvas) => {
    canvas.width = width;
    canvas.height = height;
  });
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#ff8a8a" : "#9ba3af";
}

async function runDenoise() {
  if (!device || !computePipeline || !imageBitmap) {
    setStatus("WebGPU not ready or no image loaded.", true);
    return;
  }

  setStatus("Processing robustness filter...");
  processBtn.disabled = true;

  const width = imageBitmap.width;
  const height = imageBitmap.height;

  const inputTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });

  const outputTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: inputTexture },
    { width, height, depthOrArrayLayers: 1 }
  );

  const radius = parseInt(radiusEl.value, 10);
  const blend = parseFloat(blendEl.value);
  const params = new ArrayBuffer(16);
  const view = new DataView(params);
  view.setUint32(0, radius, true);
  view.setUint32(4, 0, true);
  view.setFloat32(8, blend, true);
  view.setFloat32(12, 0, true);
  device.queue.writeBuffer(uniformBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: inputTexture.createView() },
      { binding: 1, resource: outputTexture.createView() },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(width / 8),
    Math.ceil(height / 8)
  );
  pass.end();

  const bytesPerPixel = 4;
  const paddedBytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
  const readBuffer = device.createBuffer({
    size: paddedBytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  encoder.copyTextureToBuffer(
    { texture: outputTexture },
    { buffer: readBuffer, bytesPerRow: paddedBytesPerRow },
    { width, height, depthOrArrayLayers: 1 }
  );

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await readBuffer.mapAsync(GPUMapMode.READ);
  const copy = new Uint8Array(readBuffer.getMappedRange());
  const data = new Uint8ClampedArray(width * height * 4);

  for (let row = 0; row < height; row++) {
    const srcStart = row * paddedBytesPerRow;
    const srcEnd = srcStart + width * bytesPerPixel;
    const dstStart = row * width * bytesPerPixel;
    data.set(copy.slice(srcStart, srcEnd), dstStart);
  }

  readBuffer.unmap();

  // If GPU readback returns zeros, re-run via CPU spatial box blur + blend.
  const nonZero = data.some((v) => v !== 0);
  let finalData = data;
  if (!nonZero) {
    console.warn("GPU output was empty; falling back to CPU blur.");
    finalData = cpuDenoiseCPU(imageBitmap, radius, blend);
    setStatus("GPU output empty; CPU fallback applied for analysis.", true);
  }

  const imageData = new ImageData(finalData, width, height);
  filteredCtx.putImageData(imageData, 0, 0);

  setStatus("Done. Review filtered output for watermark persistence.");
  processBtn.disabled = false;
}

function cpuDenoiseCPU(bitmap, radius, blend) {
  const w = bitmap.width;
  const h = bitmap.height;
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const ctx = tmp.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const orig = ctx.getImageData(0, 0, w, h);
  const out = new Uint8ClampedArray(orig.data);

  const idx = (x, y) => (y * w + x) * 4;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let rSum = 0, gSum = 0, bSum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = Math.min(h - 1, Math.max(0, y + dy));
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = Math.min(w - 1, Math.max(0, x + dx));
          const i = idx(xx, yy);
          rSum += orig.data[i];
          gSum += orig.data[i + 1];
          bSum += orig.data[i + 2];
          count++;
        }
      }
      const i = idx(x, y);
      const rBlur = rSum / count;
      const gBlur = gSum / count;
      const bBlur = bSum / count;
      out[i] = orig.data[i] * (1 - blend) + rBlur * blend;
      out[i + 1] = orig.data[i + 1] * (1 - blend) + gBlur * blend;
      out[i + 2] = orig.data[i + 2] * (1 - blend) + bBlur * blend;
      out[i + 3] = orig.data[i + 3];
    }
  }
  return out;
}
