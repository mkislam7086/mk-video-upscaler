const $=s=>document.querySelector(s);
const input=$("#videoInput"),drop=$("#dropZone"),fileMeta=$("#fileMeta"),processBtn=$("#processBtn"),statusEl=$("#status"),badge=$("#engineBadge"),progressWrap=$("#progressWrap"),progressBar=$("#progressBar"),percent=$("#percent"),preview=$("#preview"),beforeVideo=$("#beforeVideo"),afterVideo=$("#afterVideo"),downloadBtn=$("#downloadBtn"),resultText=$("#resultText"),previewSub=$("#previewSub");
const MODEL_URL="https://huggingface.co/Saimon8420/realesr-general-x4v3-web/resolve/main/realesr-general-x4v3-static384.onnx";
const MODEL_SIZE=384,SCALE=4,MAX_SECONDS=30,MAX_LONG_EDGE=2160;
let file=null,session=null,busy=false,mediaReady=false,graphCapture=false;
let aiInputCanvas=null,aiInputCtx=null,aiFullCanvas=null,aiFullCtx=null,aiImageData=null;
const setProgress=v=>{v=Math.max(0,Math.min(100,Math.round(v)));progressBar.style.width=v+"%";percent.textContent=v+"%"};
const status=t=>statusEl.textContent=t;
const err=e=>{console.error(e);status("Error: "+(e?.message||e));processBtn.disabled=!file||busy};
function selectedMode(){return document.querySelector('input[name="mode"]:checked').value}
function formatBytes(n){if(n<1024*1024)return (n/1024).toFixed(0)+" KB";return (n/1024/1024).toFixed(1)+" MB"}
function activateModeUI(){document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('active',x.querySelector('input').checked))}
document.querySelectorAll('input[name="mode"]').forEach(x=>x.addEventListener('change',activateModeUI));
drop.onclick=()=>input.click();
drop.ondragover=e=>e.preventDefault();
drop.ondrop=e=>{e.preventDefault();if(e.dataTransfer.files[0]){input.files=e.dataTransfer.files;input.dispatchEvent(new Event('change'))}};

function setupBuffers(){
  aiInputCanvas=document.createElement('canvas');aiInputCanvas.width=MODEL_SIZE;aiInputCanvas.height=MODEL_SIZE;
  aiInputCtx=aiInputCanvas.getContext('2d',{willReadFrequently:true});
  aiFullCanvas=document.createElement('canvas');aiFullCanvas.width=MODEL_SIZE*SCALE;aiFullCanvas.height=MODEL_SIZE*SCALE;
  aiFullCtx=aiFullCanvas.getContext('2d');
  aiImageData=aiFullCtx.createImageData(MODEL_SIZE*SCALE,MODEL_SIZE*SCALE);
}
async function init(){
  try{
    if(!navigator.gpu)throw Error("WebGPU is not available. Use recent Chrome/Edge.");
    const adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
    if(!adapter)throw Error("No compatible GPU found.");
    badge.textContent="GPU READY";status("Loading AI model…");setupBuffers();
    // Graph capture is intentionally disabled in V2.1 for browser compatibility.
    // ONNX Runtime requires fixed external GPU buffers for graph capture; feeding
    // normal CPU tensors (as this mobile pipeline does) causes the
    // "External buffer must be provided" error. The model remains fully on WebGPU.
    const base={executionProviders:["webgpu"],graphOptimizationLevel:"all"};
    session=await ort.InferenceSession.create(MODEL_URL,base);
    graphCapture=false;
    badge.textContent="AI READY · WEBGPU";
    status("AI ready · WebGPU optimized path.");
    processBtn.disabled=!file;
  }catch(e){badge.textContent="GPU ERROR";err(e)}
}
function tensorFromInput(){
  const d=aiInputCtx.getImageData(0,0,MODEL_SIZE,MODEL_SIZE).data,p=MODEL_SIZE*MODEL_SIZE,a=new Float32Array(3*p);
  for(let i=0,j=0;i<p;i++,j+=4){a[i]=d[j]/255;a[p+i]=d[j+1]/255;a[2*p+i]=d[j+2]/255}
  return new ort.Tensor('float32',a,[1,3,MODEL_SIZE,MODEL_SIZE]);
}
async function aiFrame(sourceCanvas,outputCanvas,tw,th){
  aiInputCtx.fillStyle='#000';aiInputCtx.fillRect(0,0,MODEL_SIZE,MODEL_SIZE);
  const s=Math.min(MODEL_SIZE/tw,MODEL_SIZE/th),w=Math.max(1,Math.round(tw*s)),h=Math.max(1,Math.round(th*s));
  aiInputCtx.drawImage(sourceCanvas,0,0,tw,th,(MODEL_SIZE-w)/2,(MODEL_SIZE-h)/2,w,h);
  const r=await session.run({[session.inputNames[0]]:tensorFromInput()});
  const t=r[session.outputNames[0]],H=t.dims[2],W=t.dims[3],data=t.data,p=H*W;
  if(W!==MODEL_SIZE*SCALE||H!==MODEL_SIZE*SCALE)throw Error(`Unexpected AI output ${W}×${H}.`);
  const rgba=aiImageData.data;
  for(let i=0,j=0;i<p;i++,j+=4){rgba[j]=Math.max(0,Math.min(255,Math.round(data[i]*255)));rgba[j+1]=Math.max(0,Math.min(255,Math.round(data[p+i]*255)));rgba[j+2]=Math.max(0,Math.min(255,Math.round(data[2*p+i]*255)));rgba[j+3]=255}
  aiFullCtx.putImageData(aiImageData,0,0);
  const cropX=((MODEL_SIZE-w)/2)*SCALE,cropY=((MODEL_SIZE-h)/2)*SCALE,cropW=w*SCALE,cropH=h*SCALE;
  const outCtx=outputCanvas.getContext('2d');
  outCtx.drawImage(aiFullCanvas,cropX,cropY,cropW,cropH,0,0,outputCanvas.width,outputCanvas.height);
}
async function detailFrame(src,canvas,W,H){
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const tile=384,step=352;
  const tileCanvas=document.createElement('canvas'),tileCtx=tileCanvas.getContext('2d');
  for(let y=0;y<H;y+=step)for(let x=0;x<W;x+=step){
    const tw=Math.min(tile,W-x),th=Math.min(tile,H-y);tileCanvas.width=tw;tileCanvas.height=th;tileCtx.drawImage(src,x,y,tw,th,0,0,tw,th);
    const tmp=document.createElement('canvas');tmp.width=tw*4;tmp.height=th*4;await aiFrame(tileCanvas,tmp,tw,th);
    const dx=x*(canvas.width/W),dy=y*(canvas.height/H),dw=tw*(canvas.width/W),dh=th*(canvas.height/H);ctx.drawImage(tmp,0,0,tmp.width,tmp.height,dx,dy,dw,dh);
  }
}
function updateProgress(done,total){setProgress(12+(done/Math.max(1,total))*83)}
async function processVideo(){
  if(!file||!session||busy)return;busy=true;processBtn.disabled=true;progressWrap.classList.remove('hidden');setProgress(0);preview.classList.remove('hidden');downloadBtn.classList.add('hidden');resultText.textContent="";
  let inputMedia=null,beforeUrl=null;
  try{
    const MB=window.MKMedia;inputMedia=new MB.Input({formats:MB.ALL_FORMATS,source:new MB.BlobSource(file)});
    const vt=await inputMedia.getPrimaryVideoTrack();if(!vt)throw Error("No video track found.");if(!(await vt.canDecode()))throw Error("This video codec cannot be decoded by this browser.");
    const duration=await vt.computeDuration();if(duration>MAX_SECONDS+.05)throw Error(`Video is ${duration.toFixed(1)}s. Maximum is ${MAX_SECONDS}s.`);
    const W=await vt.getDisplayWidth(),H=await vt.getDisplayHeight();
    const scale=Math.min(2,MAX_LONG_EDGE/Math.max(W,H));const outW=Math.max(2,Math.round(W*scale/2)*2),outH=Math.max(2,Math.round(H*scale/2)*2);
    const fpsInfo=await vt.computeFrameRateMetrics();const fps=Math.min(60,Math.max(1,Number(fpsInfo.averageFrameRate||30)));if(!Number.isFinite(fps))throw Error('Could not determine a valid frame rate.');
    previewSub.textContent=`${W}×${H} → ${outW}×${outH} · ${fps.toFixed(1)} fps`;
    beforeUrl=URL.createObjectURL(file);beforeVideo.src=beforeUrl;
    status("Preparing hardware video encoder…");
    const target=new MB.BufferTarget();const output=new MB.Output({format:new MB.Mp4OutputFormat({fastStart:'in-memory'}),target});
    const canvas=document.createElement('canvas');canvas.width=outW;canvas.height=outH;const ctx=canvas.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,outW,outH);
    const source=new MB.CanvasSource(canvas,{codec:'avc',quality:new MB.Quality('medium')});output.addVideoTrack(source,{frameRate:fps});
    const at=await inputMedia.getPrimaryAudioTrack();let audioSource=null;
    if(at&&await at.canDecode()){audioSource=new MB.AudioSampleSource({codec:'aac',quality:new MB.Quality('medium')});output.addAudioTrack(audioSource)}
    await output.start();
    if(audioSource){status("Copying original audio…");const asink=new MB.AudioSampleSink(at);for await(const sample of asink.samples(0,duration)){await audioSource.add(sample);sample.close()}audioSource.close()}
    const mode=selectedMode();const interval=mode==='turbo'?3:mode==='fast'?2:1;
    const total=Math.max(1,Math.ceil(duration*fps));let count=0,aiCount=0,lastAi=-1;
    const src=document.createElement('canvas');src.width=W;src.height=H;const srcCtx=src.getContext('2d');
    const vsink=new MB.VideoSampleSink(vt);
    for await(const sample of vsink.samples(0,duration)){
      const t=sample.timestamp,d=sample.duration;sample.draw(srcCtx,0,0,W,H);
      const shouldAI=(count%interval===0)||lastAi<0;
      if(shouldAI){
        status(mode==='detail'?`AI tile-enhancing frame ${count+1}/${total}…`:`AI enhancing key frame ${aiCount+1}…`);
        if(mode==='detail'){await detailFrame(src,canvas,outW,outH); } else { await aiFrame(src,canvas,W,H); } aiCount++;lastAi=count;
      }
      await source.add(t,d);sample.close();count++;updateProgress(count,total);
      await new Promise(r=>setTimeout(r,0));
    }
    source.close();await output.finalize();setProgress(100);
    const blob=new Blob([output.target.buffer],{type:'video/mp4'});const url=URL.createObjectURL(blob);afterVideo.src=url;downloadBtn.href=url;downloadBtn.classList.remove('hidden');
    const saved=Math.max(0,Math.round((1-aiCount/Math.max(1,count))*100));
    resultText.textContent=`Done · ${formatBytes(blob.size)} · ${aiCount}/${count} frames AI-processed · ${saved}% fewer AI runs · local only.`;
    status("Upscaling complete. No video was uploaded.");preview.scrollIntoView({behavior:'smooth',block:'start'});inputMedia.dispose();inputMedia=null;
  }catch(e){try{inputMedia?.dispose()}catch{}err(e)}finally{if(beforeUrl){}busy=false;processBtn.disabled=!file}
}
input.addEventListener('change',async()=>{
  file=input.files?.[0]||null;processBtn.disabled=true;preview.classList.add('hidden');downloadBtn.classList.add('hidden');if(!file)return;
  try{const v=document.createElement('video');v.preload='metadata';v.muted=true;v.playsInline=true;v.src=URL.createObjectURL(file);await new Promise((res,rej)=>{v.onloadedmetadata=res;v.onerror=()=>rej(Error('Could not read video metadata. Try MP4/H.264.'))});
    if(v.duration>MAX_SECONDS+.05){fileMeta.textContent=`${file.name} · ${v.videoWidth}×${v.videoHeight} · ${v.duration.toFixed(1)}s · too long`;fileMeta.classList.remove('hidden');status(`Maximum length is ${MAX_SECONDS} seconds.`);return}
    fileMeta.textContent=`${file.name} · ${v.videoWidth}×${v.videoHeight} · ${v.duration.toFixed(1)}s · ${formatBytes(file.size)}`;fileMeta.classList.remove('hidden');if(!session)await init();processBtn.disabled=!session;status(session?'Ready · choose Turbo for maximum speed.':'AI engine not ready.');
  }catch(e){err(e)}
});
processBtn.addEventListener('click',()=>processVideo().catch(err));
window.addEventListener('mk-media-ready',()=>{mediaReady=true;if(file&&!session)init()});
if(window.MKMedia)init();
