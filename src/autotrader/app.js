"use strict";
const el=id=>document.getElementById(id);
const money=x=>x==null?"—":"₹"+Number(x).toLocaleString("en-IN",{maximumFractionDigits:2});
const put=(id,v)=>{el(id).textContent=v;};
const cell=(tr,v)=>{const td=document.createElement("td");td.textContent=v==null?"—":String(v);tr.append(td);return td;};
let selected=null,watchlist=[],searchResults=[];
function showTab(config){el("config-panel").hidden=!config;el("monitor-panel").hidden=config;}
el("tab-monitor").onclick=()=>showTab(false);
el("tab-config").onclick=()=>showTab(true);
async function post(path,payload={}){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw Error(d.error||"Request failed");return d;
}
async function control(path,message,target="trade-control-result",payload={}){
  if(message&&!confirm(message))return;
  try{const result=await post(path,payload);put(target,JSON.stringify(result));await poll();}
  catch(e){put(target,"Error: "+e.message);}
}
el("start-trading").onclick=()=>control("/api/start");
el("stop-trading").onclick=()=>control("/api/stop","Stop new paper BUYs? Existing positions will remain open and monitored.");
el("kill-trading").onclick=()=>control("/api/emergency","Emergency halt and attempt to exit ALL paper positions.");
el("save-symbols").onclick=async()=>{
  if(!selected){put("config-result","Select an exact instrument from search results first.");return;}
  await control("/api/watchlist",null,"config-result",{action:"add",symbol:selected.symbol});
  selected=null;el("symbols").value="";el("symbol-results").replaceChildren();
};
el("symbols").addEventListener("input",async()=>{
  selected=null;const q=el("symbols").value.trim();const list=el("symbol-results");list.replaceChildren();
  if(q.length<2)return;
  try{
    const r=await fetch("/api/instruments?q="+encodeURIComponent(q),{cache:"no-store"});
    const results=await r.json();
    if(el("symbols").value.trim()!==q)return;
    searchResults=results;
    for(const item of results){
      const b=document.createElement("button");b.type="button";
      b.textContent=item.symbol+" · "+item.name+" ("+item.source+")";
      b.onclick=()=>{selected=item;el("symbols").value=item.symbol;list.replaceChildren();put("config-result","Selected "+item.symbol+(item.security_id?" · Dhan ID "+item.security_id:" · demo catalog only"));};
      list.append(b);
    }
    if(!results.length)list.textContent="No verified matches. Load a Dhan instrument master CSV to expand the catalog.";
  }catch(e){list.textContent="Instrument search unavailable: "+e.message;}
});
function render(d){
  watchlist=d.watchlist;
  put("cash",money(d.cash));put("pnl",money(d.daily.realized_pnl));
  put("positions-count",d.daily.open_positions);put("entries",d.entries_today+" / 5");
  put("halt",d.daily.halted?"HALTED":"READY");
  el("halt").title=d.daily.halt_reason||"";
  el("halt").className=d.daily.halted?"bad":"good";
  put("errors",d.errors);put("daily-trades",d.daily.trades);
  put("winrate",Number(d.daily.win_rate).toFixed(1)+"%");
  put("weekly",money(d.weekly.realized_pnl));put("unrealized",money(d.daily.unrealized_pnl));
  const watched=el("watchlist");watched.replaceChildren();
  for(const s of d.watchlist){
    const q=d.quotes[s]||{},p=d.positions[s],tr=document.createElement("tr");
    [s,money(q.price),q.volume,q.oi,p?.entry,p?.stop,p?.target,p?"OPEN":q.price?"WATCHING":"NO DATA FEED"].forEach(x=>cell(tr,x));
    watched.append(tr);
  }
  const open=el("open");open.replaceChildren();
  for(const [s,p] of Object.entries(d.positions)){
    const tr=document.createElement("tr"),price=d.quotes[s]?.price;
    [s,p.quantity,p.entry,p.stop,p.target,price?money((Number(price)-Number(p.entry))*p.quantity):"NO QUOTE"].forEach(x=>cell(tr,x));
    const td=document.createElement("td"),b=document.createElement("button");b.type="button";b.textContent="EXIT";
    b.onclick=()=>control("/api/exit","Sell "+s+" at its latest observed PAPER quote?","trade-control-result",{symbol:s});
    td.append(b);tr.append(td);open.append(tr);
  }
  if(!open.children.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=7;td.textContent="No open paper positions";tr.append(td);open.append(tr);}
  const list=el("current-symbols");list.replaceChildren();
  for(const s of d.watchlist){
    const b=document.createElement("button");b.type="button";b.textContent=s+" × Remove";
    b.onclick=()=>control("/api/watchlist","Remove "+s+" from watchlist?","config-result",{action:"remove",symbol:s});
    list.append(b);
  }
  const events=el("events");events.replaceChildren();
  for(const item of d.recent_events.slice(-8).reverse()){
    const div=document.createElement("div");div.className="event";const e=item.event||{};
    div.textContent=item.error?item.symbol+": "+item.error:item.symbol+" · "+(e.reason||"PAPER BUY")+" · Qty "+e.quantity+" · "+(e.exit?"Exit "+money(e.exit):"Entry "+money(e.entry))+(e.pnl?" · P&L "+money(e.pnl):"");
    events.append(div);
  }
  if(!events.children.length)events.textContent="No events yet.";
  const stamps=Object.values(d.quotes).map(q=>q.updated_at).filter(Boolean);
  put("updated",stamps.length?"Last quote: "+new Date(stamps.sort().at(-1)).toLocaleTimeString("en-IN",{timeZone:"Asia/Kolkata"}):"No quotes yet");
  put("connection","CONNECTED · PAPER");
}
async function poll(){
  try{const r=await fetch("/api/status",{cache:"no-store"});if(!r.ok)throw Error("HTTP "+r.status);render(await r.json());}
  catch(e){put("connection","DISCONNECTED · "+e.message);}
}
poll();setInterval(poll,1500);
