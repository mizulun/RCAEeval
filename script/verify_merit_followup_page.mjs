// Real Chrome checks using its local DevTools protocol; no npm dependencies.
import {spawn} from 'node:child_process';
import {mkdtemp,writeFile,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const root=resolve(process.argv[2]);
const profile=await mkdtemp(join(tmpdir(),'merit-viewer-'));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
 '--headless','--disable-gpu','--no-sandbox','--disable-background-networking',
 '--disable-component-update','--disable-sync','--disable-default-apps','--no-first-run',
 '--no-default-browser-check','--password-store=basic',`--user-data-dir=${profile}`,
 '--remote-debugging-port=0','about:blank'],{stdio:['ignore','ignore','pipe']});
let buffer='';
const endpoint=await new Promise((resolve,reject)=>{
 const timer=setTimeout(()=>reject(new Error('Chrome startup timed out')),30000);
 chrome.stderr.on('data',b=>{buffer+=b.toString();let m=buffer.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1]);}});
 chrome.on('error',reject);chrome.on('exit',code=>{clearTimeout(timer);reject(new Error('Chrome exited '+code));});
});
let socket;
try{
 const origin=new URL(endpoint).origin.replace('ws:','http:');
 let targets=await (await fetch(origin+'/json/list')).json();
 socket=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j;});
 let serial=0;const pending=new Map();let browserErrors=[];
 socket.onmessage=e=>{const p=JSON.parse(e.data);if(p.id){let c=pending.get(p.id);pending.delete(p.id);if(p.error)c.reject(p.error);else c.resolve(p.result);}else if(p.method==='Runtime.exceptionThrown')browserErrors.push(p.params);};
 const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{let r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await send('Runtime.enable');await send('Page.enable');
 await send('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:pathToFileURL(join(root,'index.html')).href});
 for(let i=0;i<100;i++){if(await evaluate('document.body?.dataset.ready')==='true')break;await new Promise(r=>setTimeout(r,100));}
 assert.equal(await evaluate('document.body.dataset.ready'),'true');
 assert.equal(await evaluate('document.querySelectorAll("#dataset option").length'),6);
 let checked=0;
 for(const dataset of ['re2-ss','re2-ob','re2-tt','re1-ob','re1-ss','re1-tt']){
  await evaluate(`$('dataset').value=${JSON.stringify(dataset)};$('dataset').dispatchEvent(new Event('change'));`);
  let groups=await evaluate(`[...$('group').options].map(o=>o.value)`);
  assert.equal(groups.length,dataset.startsWith('re2')?5:1);
  for(const group of groups){
   await evaluate(`$('group').value=${JSON.stringify(group)};$('group').dispatchEvent(new Event('change'));`);
   let conditions=await evaluate(`[...$('condition').options].map(o=>o.value)`);
   for(const condition of conditions){
    await evaluate(`$('condition').value=${JSON.stringify(condition)};$('condition').dispatchEvent(new Event('change'));`);
    assert.ok(await evaluate(`$('resultTable').querySelectorAll('tbody tr').length>0`));
    for(const status of ['fixed','broken','problem','ALL']){
     await evaluate(`$('caseState').value=${JSON.stringify(status)};$('caseState').dispatchEvent(new Event('change'));`);
     assert.equal(await evaluate(`$('caseTable').querySelectorAll('tbody tr').length`),Math.min(100,await evaluate('filteredCases().length')));
    }
    checked++;
   }
  }
 }
 await evaluate(`$('dataset').value='re2-ss';$('dataset').dispatchEvent(new Event('change'));$('group').value='scale';$('group').dispatchEvent(new Event('change'));$('search').value='cpu';$('search').dispatchEvent(new Event('input'));`);
 assert.ok(await evaluate(`filteredCases().every(c=>c.case.includes('cpu'))`));
 await evaluate(`$('fault').value='cpu';$('fault').dispatchEvent(new Event('change'));$('caseArm').value='fallback_only';$('caseArm').dispatchEvent(new Event('change'));`);
 assert.ok(await evaluate(`filteredCases().length>0&&filteredCases().every(c=>c.fault==='cpu'&&c.arm==='fallback_only')`));
 for(const metric of ['AC@1','AC@3','AC@5','AVG@5']){
  await evaluate(`$('metric').value=${JSON.stringify(metric)};$('metric').dispatchEvent(new Event('change'));`);
  assert.deepEqual(await evaluate(`[...document.querySelectorAll('.chart-row strong')].map(x=>x.textContent)`),
                   await evaluate(`selected().map(r=>pct(r.metrics?.[${JSON.stringify(metric)}]?.value))`));
 }
 let csv=await evaluate(`(async()=>{let saved,blob;let old=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;URL.createObjectURL=b=>{blob=b;return 'blob:test'};HTMLAnchorElement.prototype.click=function(){saved=this.download};$('download').click();URL.createObjectURL=old;HTMLAnchorElement.prototype.click=click;return {name:saved,text:await blob.text(),n:filteredCases().length};})()`);
 assert.equal(csv.name,'merit-filtered-cases.csv');assert.equal(csv.text.trim().split('\n').length,csv.n+1);
 await evaluate(`$('search').value='';$('fault').value='ALL';$('fault').dispatchEvent(new Event('change'));$('caseArm').value='ALL';$('caseArm').dispatchEvent(new Event('change'));$('metric').value='AC@1';$('metric').dispatchEvent(new Event('change'));`);
 let shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(join(root,'viewer-desktop.png'),Buffer.from(shot.data,'base64'));
 await evaluate(`$('dataset').scrollIntoView({block:'start'});`);
 shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(join(root,'viewer-results.png'),Buffer.from(shot.data,'base64'));
 await evaluate('window.scrollTo(0,0)');
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 assert.ok(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'));
 shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(join(root,'viewer-mobile.png'),Buffer.from(shot.data,'base64'));
 assert.equal(browserErrors.length,0);
 const report={ready:true,datasets:6,scenarioViews:checked,caseFilters:true,faultFilter:true,armFilter:true,metricCharts:true,search:true,csv:true,mobileNoPageOverflow:true,browserErrors};
 await writeFile(join(root,'browser-verification.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
}finally{socket?.close();chrome.kill('SIGTERM');}
