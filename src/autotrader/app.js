"use strict";
const el=id=>document.getElementById(id);
const fmt=x=>x==null?"—":String(x);
const money=x=>x==null?"—":"₹"+Number(x).toLocaleString("en-IN",{maximumFractionDigits:2});
const put=(id,value)=>{el(id).textContent=value;};
const cell=(row,value)=>{const td=document.createElement("td");td.textContent=fmt(value);row.append(td);};
function render(data){
  put("cash",money(data.cash));
  put("pnl",money(data.daily.realized_pnl));
  put("positions-count",data.daily.open_positions);
  put("entries",data.entries_today+" / 5");
  put("halt",data.daily.halted?"HALTED":"READY");
  el("halt").title=data.daily.halt_reason||"";
  el("halt").className=data.daily.halted?"bad":"good";
  put("errors",data.errors);
  put("daily-trades",data.daily.trades);
  put("winrate",Number(data.daily.win_rate).toFixed(1)+"%");
  put("weekly",money(data.weekly.realized_pnl));
  put("unrealized",money(data.daily.unrealized_pnl));
  const tbody=el("watchlist");tbody.replaceChildren();
  for(const symbol of data.watchlist){
    const q=data.quotes[symbol]||{},p=data.positions[symbol];
    const tr=document.createElement("tr");
    [symbol,money(q.price),q.volume,q.oi,p?.entry,p?.stop,p?.target,p?"OPEN":"WATCHING"].forEach(x=>cell(tr,x));
    tbody.append(tr);
  }
  const open=el("open");open.replaceChildren();
  for(const [symbol,p] of Object.entries(data.positions)){
    const tr=document.createElement("tr");
    [symbol,p.quantity,p.entry,p.stop,p.target,money((Number(data.quotes[symbol]?.price||p.entry)-Number(p.entry))*Number(p.quantity))].forEach(x=>cell(tr,x));open.append(tr);
  }
  if(!open.children.length){const tr=document.createElement("tr");const td=document.createElement("td");td.colSpan=6;td.textContent="No open paper positions";tr.append(td);open.append(tr);}
  const events=el("events");events.replaceChildren();
  for(const item of data.recent_events.slice(-8).reverse()){
    const div=document.createElement("div");div.className="event";
    const e=item.event||{};
    div.textContent=item.error?item.symbol+": "+item.error:item.symbol+" · "+(e.reason||"PAPER BUY")+" · Qty "+e.quantity+" · "+(e.exit?"Exit "+money(e.exit):"Entry "+money(e.entry))+(e.pnl?" · P&L "+money(e.pnl):"");
    events.append(div);
  }
  if(!events.children.length)events.textContent="No events yet.";
  const stamps=Object.values(data.quotes).map(q=>q.updated_at).filter(Boolean);
  put("updated",stamps.length?"Last quote: "+new Date(stamps.sort().at(-1)).toLocaleTimeString("en-IN",{timeZone:"Asia/Kolkata"}):"No quotes yet");
  put("connection","CONNECTED · PAPER");
}
async function poll(){
  try{const response=await fetch("/api/status",{cache:"no-store"});if(!response.ok)throw Error("HTTP "+response.status);render(await response.json());}
  catch(e){put("connection","DISCONNECTED · "+e.message);}
}
poll();setInterval(poll,1500);

let lastSymbols="";
function showTab(config){
  el("config-panel").hidden=!config;
  el("monitor-panel").hidden=config;
}
el("tab-monitor").addEventListener("click",()=>showTab(false));
el("tab-config").addEventListener("click",()=>showTab(true));
async function post(path,payload){
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data=await response.json();
  if(!response.ok)throw Error(data.error||"Request failed");
  return data;
}
el("save-symbols").addEventListener("click",async()=>{
  try{
    const symbols=el("symbols").value.split(",").map(s=>s.trim().toUpperCase()).filter(Boolean);
    const result=await post("/api/watchlist",{symbols});
    el("config-result").textContent="Saved: "+result.watchlist.join(", ");
    lastSymbols=result.watchlist.join(", ");
  }catch(e){el("config-result").textContent="Error: "+e.message;}
});
el("emergency").addEventListener("click",async()=>{
  if(!confirm("HALT all new paper BUYs and attempt to SELL ALL paper positions at last observed prices?"))return;
  try{
    const result=await post("/api/emergency",{});
    el("emergency-result").textContent="Closed: "+result.closed.length+"; remaining: "+result.remaining.join(", ")+"; errors: "+JSON.stringify(result.errors);
  }catch(e){el("emergency-result").textContent="Error: "+e.message;}
});
const oldRender=render;
render=function(data){
  oldRender(data);
  const symbols=data.watchlist.join(", ");
  if(document.activeElement!==el("symbols") && lastSymbols!==symbols){
    el("symbols").value=symbols;
    lastSymbols=symbols;
  }
};

async function control(path,confirmation){
  if(confirmation && !confirm(confirmation))return;
  try{
    const result=await post(path,{});
    el("trade-control-result").textContent=JSON.stringify(result);
    await poll();
  }catch(e){el("trade-control-result").textContent="Error: "+e.message;}
}
el("start-trading").addEventListener("click",()=>control("/api/start"));
el("stop-trading").addEventListener("click",()=>control("/api/stop","STOP new paper BUYs and attempt to exit ALL open paper positions?"));
el("kill-trading").addEventListener("click",()=>control("/api/emergency","EMERGENCY KILL: permanently halt paper entries and attempt to exit all positions? START cannot override this."));
