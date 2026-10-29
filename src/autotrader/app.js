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
    [symbol,p.quantity,p.entry,p.stop,p.target].forEach(x=>cell(tr,x));open.append(tr);
  }
  if(!open.children.length){const tr=document.createElement("tr");const td=document.createElement("td");td.colSpan=5;td.textContent="No open paper positions";tr.append(td);open.append(tr);}
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
