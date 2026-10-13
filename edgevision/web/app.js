const $=(id)=>document.getElementById(id);
const pretty=(x)=>Number(x).toLocaleString();
const label=(x)=>String(x||'').replace(/^./,c=>c.toUpperCase());
const text=(id,value)=>{if($(id))$(id).textContent=String(value)};
const clock=()=>text('clock',new Date().toLocaleTimeString('en-GB',{hour12:false}));
clock();setInterval(clock,1000);
let configLoaded=false;
let running=true;
let previousEvents='';

async function getJson(url,opts){const r=await fetch(url,{cache:'no-store',...opts});if(!r.ok)throw new Error('HTTP '+r.status);return r.json()}
function eventNode(item){
 const el=document.createElement('div');el.className='event-item';
 const icon=document.createElement('span');icon.className='event-icon';icon.textContent='⌗';
 const body=document.createElement('div');body.className='event-content';
 const title=document.createElement('div');title.className='event-title';title.textContent=label(item.label);
 const meta=document.createElement('div');meta.className='event-meta';
 const stamp=document.createElement('span');stamp.textContent=new Date(item.confirmed_at*1000).toLocaleTimeString();
 const tr=document.createElement('span');tr.textContent=`TRACK #${String(item.track_id).padStart(3,'0')}`;
 const src=document.createElement('span');src.textContent=item.source.includes('heuristic')?'HEURISTIC CANDIDATE':`${Math.round(item.confidence*100)}% MODEL CONF.`;
 meta.append(stamp,tr,src);body.append(title,meta);
 const pill=document.createElement('span');pill.className='severity '+item.severity;pill.textContent=item.severity;
 el.append(icon,body,pill);return el;
}
async function updateEvents(){
 const j=await getJson('/api/v1/events?limit=15');
 text('event-pill',pretty(j.total)+' EVENTS');
 const key=JSON.stringify(j.events.map(x=>x.id));
 if(key===previousEvents)return;previousEvents=key;
 const root=$('event-list');root.replaceChildren();
 if(!j.events.length){
  const empty=document.createElement('div');empty.className='empty-state';
  empty.innerHTML='<span class="empty-mark">⌁</span><strong>Waiting for verified events</strong><p>Confirmed observations will appear here automatically.</p>';
  root.append(empty);
 } else j.events.forEach(e=>root.append(eventNode(e)));
}
async function updateTelemetry(){
 try{
  const s=await getJson('/api/v1/telemetry');
  running=!s.error;
  text('fps',s.throughput_fps.toFixed(1));text('latency',s.processing_ms_p95.toFixed(1));
  text('tracks',pretty(s.active_tracks));text('event-count',pretty(s.events));
  text('frame-label','FRAME '+String(s.frames).padStart(6,'0'));
  text('source-tag',s.source.toUpperCase());text('source-detail',s.source);
  text('captured',pretty(s.captured_frames));text('dropped',pretty(s.dropped_frames));
  text('queue',pretty(s.queue_depth));text('median',s.processing_ms_p50.toFixed(2)+' ms');
  text('health',s.error?'DEGRADED':s.stream_ended?'SOURCE ENDED':'OPERATIONAL');
  text('health-tag',s.error?'DEGRADED':s.stream_ended?'SOURCE ENDED':'LOCAL / ACTIVE');
  text('runtime-mode',s.mode==='heuristic demo'?'CANDIDATE DEMO':'TFLITE INFERENCE');
  text('mode-badge',s.mode==='heuristic demo'?'HEURISTIC DEMO':'TFLITE MODEL');
  if(s.mode!=='heuristic demo'){
    $('model-warning').textContent='TFLite model active. Low-confidence or ambiguous classifications are withheld from verified alerts. Model performance depends on your training and calibration.';
  }
  if(!configLoaded){$('conf').value=Math.round(s.config.confidence_threshold*100);$('hits').value=s.config.min_track_hits;changeLabels();configLoaded=true}
  await updateEvents();
 }catch(error){text('health','API UNAVAILABLE');text('health-tag','OFFLINE');console.warn('EdgeVision telemetry:',error.message)}
}
function changeLabels(){text('conf-out',$('conf').value+'%');text('hits-out',$('hits').value+' FRAMES')}
$('conf').addEventListener('input',changeLabels);$('hits').addEventListener('input',changeLabels);
$('apply').addEventListener('click',async()=>{
 const btn=$('apply');btn.disabled=true;text('save-msg','Applying…');
 try{await getJson('/api/v1/config',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({confidence_threshold:Number($('conf').value)/100,min_track_hits:Number($('hits').value)})});text('save-msg','✓ Saved to live runtime')}
 catch(e){text('save-msg','Could not save · '+e.message)}
 finally{btn.disabled=false}
});
const navLinks=Array.from(document.querySelectorAll('.nav-stack a'));
navLinks.forEach(link=>link.addEventListener('click',()=>{navLinks.forEach(x=>x.classList.remove('nav-active'));link.classList.add('nav-active')}));
updateTelemetry();setInterval(updateTelemetry,1250);
