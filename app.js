const videoInput = document.querySelector('#videoInput');
const videoInfo = document.querySelector('#videoInfo');
const statusEl = document.querySelector('#status');
const testBtn = document.querySelector('#testBtn');
const preview = document.querySelector('#preview');
const before = document.querySelector('#before');
const after = document.querySelector('#after');
const resultText = document.querySelector('#resultText');
const progressWrap = document.querySelector('#progressWrap');
const progressBar = document.querySelector('#progressBar');
const progressText = document.querySelector('#progressText');

let file = null;
let session = null;
const MODEL_URL = 'https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx';

function setProgress(v){
  const p=Math.max(0,Math.min(100,Math.round(v)));
  progressBar.style.width=p+'%';
  progressText.textContent=p+'%';
}

async function initGPU(){
  if (!('gpu' in navigator)) {
    statusEl.textContent='❌ WebGPU is not available in this browser.';
    return false;
  }
  try{
    const adapter=await navigator.gpu.requestAdapter();
    if(!adapter){statusEl.textContent='❌ No compatible GPU adapter found.';return false;}
    statusEl.textContent='✅ WebGPU available. AI can run locally in this browser.';
    return true;
  }catch(e){
    statusEl.textContent='❌ WebGPU check failed: '+e.message;
    return false;
  }
}

async function loadModel(){
  statusEl.textContent='Loading AI model (~5 MB)…';
  setProgress(5);
  session = await ort.InferenceSession.create(MODEL_URL,{
    executionProviders:['webgpu'],
    graphOptimizationLevel:'all'
  });
  setProgress(20);
  statusEl.textContent='✅ AI model loaded.';
}

function getVideoMeta(f){
  return new Promise((resolve,reject)=>{
    const v=document.createElement('video');
    v.preload='metadata';
    v.onloadedmetadata=()=>{resolve({duration:v.duration,width:v.videoWidth,height:v.videoHeight,fps:'unknown'}); URL.revokeObjectURL(v.src);};
    v.onerror=()=>reject(new Error('Could not read video metadata.'));
    v.src=URL.createObjectURL(f);
  });
}

function seekVideo(v,t){
  return new Promise((resolve,reject)=>{
    const done=()=>{v.removeEventListener('seeked',done);resolve();};
    v.addEventListener('seeked',done,{once:true});
    v.currentTime=t;
    setTimeout(()=>reject(new Error('Video seek timed out.')),15000);
  });
}

function imageToTensor(canvas){
  const w=canvas.width,h=canvas.height;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  const rgba=ctx.getImageData(0,0,w,h).data;
  const data=new Float32Array(3*w*h);
  for(let i=0,p=0;i<rgba.length;i+=4,p++){
    data[p]=rgba[i]/255;
    data[w*h+p]=rgba[i+1]/255;
    data[2*w*h+p]=rgba[i+2]/255;
  }
  return new ort.Tensor('float32',data,[1,3,h,w]);
}

function tensorToCanvas(t,canvas){
  const [n,c,h,w]=t.dims;
  canvas.width=w;canvas.height=h;
  const out=new Uint8ClampedArray(w*h*4);
  const data=t.data;
  for(let p=0;p<w*h;p++){
    out[p*4]=Math.max(0,Math.min(255,Math.round(data[p]*255)));
    out[p*4+1]=Math.max(0,Math.min(255,Math.round(data[w*h+p]*255)));
    out[p*4+2]=Math.max(0,Math.min(255,Math.round(data[2*w*h+p]*255)));
    out[p*4+3]=255;
  }
  canvas.getContext('2d').putImageData(new ImageData(out,w,h),0,0);
}

async function testOneFrame(){
  if(!file||!session)return;
  testBtn.disabled=true;
  progressWrap.classList.remove('hidden');
  setProgress(25);
  const v=document.createElement('video');
  v.muted=true;v.playsInline=true;v.preload='auto';
  v.src=URL.createObjectURL(file);
  await new Promise((res,rej)=>{v.onloadedmetadata=res;v.onerror=()=>rej(new Error('Video decode failed.'));});
  await seekVideo(v,Math.min(0.5,Math.max(0,v.duration/2)));
  const maxSide=640;
  const scale=Math.min(1,maxSide/Math.max(v.videoWidth,v.videoHeight));
  const w=Math.max(16,Math.round(v.videoWidth*scale));
  const h=Math.max(16,Math.round(v.videoHeight*scale));
  before.width=w;before.height=h;
  before.getContext('2d').drawImage(v,0,0,w,h);
  setProgress(40);
  statusEl.textContent='Running Real-ESRGAN on the selected frame…';
  const input=imageToTensor(before);
  const output=await session.run({input});
  setProgress(90);
  tensorToCanvas(output.output,after);
  setProgress(100);
  preview.classList.remove('hidden');
  resultText.textContent=`Frame processed locally: ${w}×${h} → ${after.width}×${after.height}.`;
  statusEl.textContent='✅ Frame AI test complete. No video was uploaded.';
  URL.revokeObjectURL(v.src);
  testBtn.disabled=false;
}

videoInput.addEventListener('change',async()=>{
  file=videoInput.files?.[0]||null;
  testBtn.disabled=true;
  preview.classList.add('hidden');
  if(!file)return;
  try{
    const m=await getVideoMeta(file);
    const limit=10;
    videoInfo.classList.remove('hidden');
    videoInfo.textContent=`${file.name} • ${m.width}×${m.height} • ${m.duration.toFixed(1)} sec`;
    if(m.duration>limit){
      statusEl.textContent=`❌ Prototype limit is ${limit} seconds. Final V1 target will be 30 seconds.`;
      return;
    }
    if(await initGPU()){
      await loadModel();
      testBtn.disabled=false;
    }
  }catch(e){
    statusEl.textContent='❌ '+e.message;
  }
});

initGPU();
