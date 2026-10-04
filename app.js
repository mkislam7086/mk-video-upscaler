const videoInput=document.querySelector("#videoInput"),videoInfo=document.querySelector("#videoInfo"),statusEl=document.querySelector("#status"),testBtn=document.querySelector("#testBtn"),preview=document.querySelector("#preview"),before=document.querySelector("#before"),after=document.querySelector("#after"),resultText=document.querySelector("#resultText"),progressWrap=document.querySelector("#progressWrap"),progressBar=document.querySelector("#progressBar"),progressText=document.querySelector("#progressText");
let file=null,session=null,gpuReady=false;

// WebGPU-safe fixed-shape Real-ESRGAN model.
// Input: 1x3x384x384 -> output: 1x3x1536x1536.
const MODEL_URL="https://huggingface.co/Saimon8420/realesr-general-x4v3-web/resolve/main/realesr-general-x4v3-static384.onnx";
const MODEL_SIZE=384, SCALE=4;

const setProgress=v=>{v=Math.max(0,Math.min(100,Math.round(v)));progressBar.style.width=v+"%";progressText.textContent=v+"%"};
const status=(s,e=false)=>statusEl.textContent=(e?"❌ ":"")+s;
function showError(e){console.error(e);const m=e?.message||String(e);status(m.slice(0,500),true);resultText.textContent="Technical error: "+m.slice(0,700);preview.classList.remove("hidden");testBtn.disabled=false}

async function initGPU(){
  if(!("gpu" in navigator)){status("WebGPU is not available in this browser.",true);return false}
  try{
    const a=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
    if(!a){status("No compatible GPU adapter found.",true);return false}
    gpuReady=true;status("WebGPU available. Loading fixed-size AI model…");return true
  }catch(e){showError(e);return false}
}

async function loadModel(){
  if(!gpuReady)throw Error("WebGPU is not ready.");
  status("Loading WebGPU-safe AI model…");setProgress(5);
  session=await ort.InferenceSession.create(MODEL_URL,{
    executionProviders:["webgpu"],
    graphOptimizationLevel:"all"
  });
  console.log("inputs",session.inputNames,"outputs",session.outputNames);
  if(!session.inputNames.length||!session.outputNames.length)throw Error("AI model loaded but no input/output nodes were found.");
  setProgress(20);status("AI model loaded. Ready for frame test.");
}

function meta(f){
  return new Promise((res,rej)=>{
    const v=document.createElement("video");v.preload="metadata";v.muted=true;v.playsInline=true;
    const cleanup=()=>{v.onloadedmetadata=null;v.onerror=null};
    v.onloadedmetadata=()=>{const x={duration:v.duration,width:v.videoWidth,height:v.videoHeight};cleanup();URL.revokeObjectURL(v.src);res(x)};
    v.onerror=()=>{cleanup();URL.revokeObjectURL(v.src);rej(Error("Could not read video metadata. Try an MP4/H.264 video."))};
    v.src=URL.createObjectURL(f)
  })
}

function loadVideo(f){
  return new Promise((res,rej)=>{
    const v=document.createElement("video");v.preload="auto";v.muted=true;v.playsInline=true;
    let settled=false;
    const fail=()=>{if(!settled){settled=true;URL.revokeObjectURL(v.src);rej(Error("Video decode failed. Try an MP4/H.264 video."))}};
    v.onerror=fail;
    v.onloadeddata=()=>{if(!settled){settled=true;res(v)}};
    v.src=URL.createObjectURL(f);
    v.load();
  })
}

function seek(v,t){
  return new Promise((res,rej)=>{
    let timer;
    const done=()=>{clearTimeout(timer);v.removeEventListener("seeked",done);res()};
    v.addEventListener("seeked",done,{once:true});
    v.currentTime=t;
    timer=setTimeout(()=>{v.removeEventListener("seeked",done);rej(Error("Video seek timed out."))},15000)
  })
}

// Put the source frame into a 384x384 letterboxed RGB canvas.
// This preserves the original aspect ratio; no stretching.
function makePaddedCanvas(v){
  const srcW=v.videoWidth,srcH=v.videoHeight;
  if(!srcW||!srcH)throw Error("Video frame has no valid dimensions.");
  const c=document.createElement("canvas");c.width=MODEL_SIZE;c.height=MODEL_SIZE;
  const ctx=c.getContext("2d");
  ctx.fillStyle="#000";ctx.fillRect(0,0,MODEL_SIZE,MODEL_SIZE);
  const s=Math.min(MODEL_SIZE/srcW,MODEL_SIZE/srcH);
  const w=Math.max(1,Math.round(srcW*s)),h=Math.max(1,Math.round(srcH*s));
  const x=Math.floor((MODEL_SIZE-w)/2),y=Math.floor((MODEL_SIZE-h)/2);
  ctx.drawImage(v,x,y,w,h);
  return {canvas:c,scaledW:w,scaledH:h,x,y,scale:s,srcW,srcH}
}

function tensorFromCanvas(canvas){
  const ctx=canvas.getContext("2d",{willReadFrequently:true}),d=ctx.getImageData(0,0,MODEL_SIZE,MODEL_SIZE).data,p=MODEL_SIZE*MODEL_SIZE,a=new Float32Array(3*p);
  for(let x=0,i=0;x<p;x++,i+=4){a[x]=d[i]/255;a[p+x]=d[i+1]/255;a[2*p+x]=d[i+2]/255}
  return new ort.Tensor("float32",a,[1,3,MODEL_SIZE,MODEL_SIZE])
}

function outputToCanvas(t,box){
  if(!t?.dims||t.dims.length!==4)throw Error("AI returned an invalid output tensor.");
  const [n,ch,h,w]=t.dims;
  if(n!==1||ch<3||h!==MODEL_SIZE*SCALE||w!==MODEL_SIZE*SCALE)throw Error("Unexpected AI output shape: "+t.dims.join(" × "));
  const d=t.data,p=h*w,o=new Uint8ClampedArray(p*4);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x;o[i*4]=Math.max(0,Math.min(255,Math.round(d[i]*255)));o[i*4+1]=Math.max(0,Math.min(255,Math.round(d[p+i]*255)));o[i*4+2]=Math.max(0,Math.min(255,Math.round(d[2*p+i]*255)));o[i*4+3]=255}
  const full=document.createElement("canvas");full.width=w;full.height=h;full.getContext("2d").putImageData(new ImageData(o,w,h),0,0);
  // Crop model padding, then resize to the original frame dimensions.
  const cropX=Math.round(box.x*SCALE),cropY=Math.round(box.y*SCALE),cropW=Math.round(box.scaledW*SCALE),cropH=Math.round(box.scaledH*SCALE);
  after.width=box.srcW*4;after.height=box.srcH*4;
  after.getContext("2d").drawImage(full,cropX,cropY,cropW,cropH,0,0,after.width,after.height);
}

async function test(){
  if(!file)throw Error("Please select a video first.");
  if(!session)throw Error("AI model is not ready yet.");
  testBtn.disabled=true;progressWrap.classList.remove("hidden");preview.classList.remove("hidden");setProgress(25);status("Opening selected video…");
  let v=null;
  try{
    v=await loadVideo(file);
    await seek(v,Math.min(.5,Math.max(0,v.duration/2)));
    const box=makePaddedCanvas(v);
    before.width=box.srcW;before.height=box.srcH;
    before.getContext("2d").drawImage(v,0,0,box.srcW,box.srcH);
    setProgress(40);status(`Running WebGPU AI on fixed ${MODEL_SIZE}×${MODEL_SIZE} input…`);
    const r=await session.run({[session.inputNames[0]]:tensorFromCanvas(box.canvas)});
    const out=r[session.outputNames[0]];
    if(!out)throw Error("AI completed but output was not returned.");
    setProgress(85);outputToCanvas(out,box);setProgress(100);
    resultText.textContent=`SUCCESS: ${box.srcW}×${box.srcH} → ${after.width}×${after.height}. Aspect ratio preserved; processed locally.`;
    status("AI frame test complete. No video was uploaded.")
  }finally{
    if(v?.src)URL.revokeObjectURL(v.src);
    testBtn.disabled=false
  }
}

videoInput.addEventListener("change",async()=>{
  file=videoInput.files?.[0]||null;testBtn.disabled=true;preview.classList.add("hidden");progressWrap.classList.add("hidden");setProgress(0);if(!file)return;
  try{
    const m=await meta(file),limit=10;
    videoInfo.classList.remove("hidden");videoInfo.textContent=`${file.name} • ${m.width}×${m.height} • ${m.duration.toFixed(1)} sec`;
    if(m.duration>limit){status(`Prototype limit is ${limit} seconds. Final V1 target will be 30 seconds.`,true);return}
    if(await initGPU()){await loadModel();testBtn.disabled=false}
  }catch(e){showError(e)}
});
testBtn.addEventListener("click",()=>test().catch(showError));
initGPU().catch(showError);
