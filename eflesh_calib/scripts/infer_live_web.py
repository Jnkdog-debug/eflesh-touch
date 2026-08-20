#!/usr/bin/env python3
"""
实时接触热力图（web 版）—— 浏览器打开 http://<orin-ip>:8899

后端(Orin): 皮肤帧 → ΔB(在线基线) → MLP → (x,y,z,Fz, Fx,Fy 剪切)，JSON 输出
前端(浏览器): canvas 热力图 —— 接触点高斯辉斑 + 衰减余晖 + 剪切箭头（青色, 指向推的方向）

Usage:
  python scripts/infer_live_web.py --artifact data/ds_v1f_mlp.pt [--http 8899]
"""

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from eflesh_calib import config
from eflesh_calib.skin_stream import SkinStats, skin_reader
from eflesh_calib.util import Latest
from infer_live import load_model

STATE = {"x": 0.0, "y": 0.0, "z": 0.0, "fz": 0.0, "fx": 0.0, "fy": 0.0,
         "mag": 0.0, "hz": 0.0, "contact": False, "t": 0.0}
LOCK = threading.Lock()

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>eFlesh 实时触觉</title>
<style>
 body{background:#111;color:#ddd;font-family:monospace;margin:0;
      display:flex;flex-direction:column;align-items:center;padding:12px}
 canvas{border:1px solid #444;border-radius:6px}
 #read{margin-top:8px;font-size:18px;letter-spacing:1px;white-space:pre}
</style></head><body>
<canvas id="cv" width="440" height="440"></canvas>
<div id="read">…</div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const N=110, PX=cv.width, mm2px=PX/40;         // 40mm 皮肤
let grid=new Float32Array(N*N), scaleF=1.0;
function cmap(v){                              // 黑→蓝→青→黄→红
 v=Math.max(0,Math.min(1,v));
 const st=[[0,0,0],[0,0,.55],[0,.8,1],[1,1,0],[1,0,0]];
 const p=v*(st.length-1),i=Math.min(Math.floor(p),st.length-2),f=p-i;
 return [0,1,2].map(k=>Math.round(255*(st[i][k]*(1-f)+st[i+1][k]*f)));
}
function draw(s){
 if(s.contact){                                // 高斯辉斑落在 (x,y)
  scaleF=Math.max(scaleF*0.999, Math.abs(s.fz), 0.3);
  const cx=(s.x+20)/40*N, cy=(20-s.y)/40*N, amp=Math.min(Math.abs(s.fz)/scaleF,1)*0.5;
  const R=9;
  for(let j=-R;j<=R;j++)for(let i=-R;i<=R;i++){
   const gx=Math.round(cx+i),gy=Math.round(cy+j);
   if(gx<0||gy<0||gx>=N||gy>=N)continue;
   grid[gy*N+gx]=Math.min(1,grid[gy*N+gx]+amp*Math.exp(-(i*i+j*j)/(R*R/3)));
  }
 }
 for(let k=0;k<N*N;k++)grid[k]*=0.955;         // 余晖衰减
 // 渲染
 const img=ctx.createImageData(N,N);
 for(let k=0;k<N*N;k++){const c=cmap(grid[k]);
  img.data[k*4]=c[0];img.data[k*4+1]=c[1];img.data[k*4+2]=c[2];img.data[k*4+3]=255;}
 const off=ctx.createImageData(N,N);off.data.set(img.data);
 ctx.imageSmoothingEnabled=true;
 const tmp=document.createElement('canvas');tmp.width=N;tmp.height=N;
 tmp.getContext('2d').putImageData(off,0,0);
 ctx.drawImage(tmp,0,0,PX,PX);
 // 网格 + 传感器位置
 ctx.strokeStyle='rgba(255,255,255,.08)';
 for(let g=0;g<=4;g++){const p=g*PX/4;
  ctx.beginPath();ctx.moveTo(p,0);ctx.lineTo(p,PX);ctx.stroke();
  ctx.beginPath();ctx.moveTo(0,p);ctx.lineTo(PX,p);ctx.stroke();}
 // 传感器标记: 实测定案(拔线+batch002 响应场验证, docs/2026-08-20-hardware-layout.md)
 // S1右下 S2左下 S3右上 S4左上(磁体反装) S5中心; 坐标为设计近似±12mm
 const SENS=[[12,-12,'S1'],[-12,-12,'S2'],[12,12,'S3'],[-12,12,'S4'],[0,0,'S5']];
 ctx.fillStyle='rgba(255,255,255,.35)';
 SENS.forEach(p=>{ctx.beginPath();
  ctx.arc((p[0]+20)*mm2px,(20-p[1])*mm2px,2.5,0,7);ctx.fill();});
 ctx.fillStyle='rgba(255,255,255,.55)';ctx.font='9px monospace';
 SENS.forEach(p=>{ctx.fillText(p[2],(p[0]+20)*mm2px+4,(20-p[1])*mm2px-4);});
 // 接触标记
 if(s.contact){
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
  const px=(s.x+20)*mm2px,py=(20-s.y)*mm2px;
  ctx.beginPath();ctx.arc(px,py,8,0,7);ctx.stroke();
  ctx.beginPath();ctx.moveTo(px-12,py);ctx.lineTo(px+12,py);
  ctx.moveTo(px,py-12);ctx.lineTo(px,py+12);ctx.stroke();
  // 剪切箭头: 方向=(Fx,Fy)皮肤系, 长度∝|F|, 屏幕y向下取反
  const fmag=Math.hypot(s.fx,s.fy);
  if(fmag>0.08){
   const L=Math.min(34, 18+30*fmag/0.7), a=Math.atan2(-s.fy,s.fx);
   const tx=px+L*Math.cos(a), ty=py+L*Math.sin(a);
   ctx.strokeStyle='#4fe';ctx.lineWidth=2.5;
   ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(tx,ty);ctx.stroke();
   ctx.beginPath();ctx.moveTo(tx,ty);          // 箭头翼
   ctx.lineTo(tx-7*Math.cos(a-0.45),ty-7*Math.sin(a-0.45));
   ctx.moveTo(tx,ty);
   ctx.lineTo(tx-7*Math.cos(a+0.45),ty-7*Math.sin(a+0.45));
   ctx.stroke();
   ctx.fillStyle='#4fe';ctx.font='11px monospace';
   ctx.fillText(`⇶ ${fmag.toFixed(2)}N`,tx+4,ty-4);
  }
 }
 const nums=s.contact?
  `  x=${s.x.toFixed(2)}mm  y=${s.y.toFixed(2)}mm  z=${s.z.toFixed(2)}mm  Fz=${s.fz.toFixed(2)}N  Fx=${s.fx.toFixed(2)} Fy=${s.fy.toFixed(2)}N`
  :'  — 未接触，预测值在训练分布外，已屏蔽 —';
 document.getElementById('read').textContent =
  (s.contact?'● 接触  ':'○ 无接触')+nums+
  `  ‖ΔB‖=${s.mag.toFixed(0)}µT  ${s.hz.toFixed(0)}Hz`;
}
async function loop(){
 try{const r=await fetch('/state');draw(await r.json());}
 catch(e){document.getElementById('read').textContent='连接断开，等后端…';}
 setTimeout(loop,60);
}
loop();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/state":
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            self.send_response(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):   # 静默访问日志
        pass


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.101.16", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--port", default=config.SKIN_PORT)
    ap.add_argument("--baud", type=int, default=config.SKIN_BAUD)
    ap.add_argument("--http", type=int, default=8899)
    ap.add_argument("--init-s", type=float, default=3.0)
    args = ap.parse_args()

    if not Path(args.port).exists():               # 串口重插可能变 ttyUSB1, 同 runtime.py 回退
        import glob
        alts = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        if not alts:
            raise SystemExit(f"串口 {args.port} 不存在, 也没找到其他 /dev/ttyUSB*")
        print(f"{args.port} 不存在, 改用 {alts[0]}")
        args.port = alts[0]

    model, x_mean, x_std, y_mean, y_std = load_model(args.artifact)
    import torch

    probe = socket.socket()                       # 先探端口:被占直接说人话退出
    try:
        probe.bind(("0.0.0.0", args.http))
    except OSError:
        raise SystemExit(f"端口 {args.http} 已被占用 —— 八成已有一个实例在跑,"
                         f"浏览器直接开 http://<Orin-IP>:{args.http} 即可;"
                         f"要重启先执行: fuser -k {args.http}/tcp")
    finally:
        probe.close()

    q: deque = deque(maxlen=2000)
    latest = Latest()
    stats = SkinStats()
    stop = threading.Event()
    threading.Thread(target=skin_reader,
                     args=(args.port, args.baud, q, stats, latest, stop),
                     daemon=True).start()

    print(f"采 {args.init_s}s 无接触基线 …")
    t0 = time.time()
    while time.time() - t0 < args.init_s and len(q) < 300:
        time.sleep(0.05)
    frames = list(q.copy())
    B0 = np.median(np.stack([f[2] for f in frames]).reshape(len(frames), -1), axis=0)

    srv = ThreadingHTTPServer(("0.0.0.0", args.http), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n★ 浏览器打开: http://{lan_ip()}:{args.http}\n")

    calm_since = None
    try:
        while True:
            v = latest.get()
            if v is None:
                time.sleep(0.005)
                continue
            _, B = v
            b = B.ravel().astype(np.float64)
            db = b - B0
            mag = float(np.linalg.norm(db))
            now = time.time()
            contact = mag >= 20.0
            if not contact:                    # 在线零点（慢 EMA）
                if calm_since is None:
                    calm_since = now
                elif now - calm_since > 2.0:
                    B0 = 0.98 * B0 + 0.02 * b
            else:
                calm_since = None
            x = (db - x_mean) / x_std
            with torch.no_grad():
                yn = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            y = yn.numpy()[0] * y_std + y_mean
            with LOCK:
                STATE.update(x=float(y[0]), y=float(y[1]), z=float(y[2]),
                             fz=float(y[3]) if len(y) > 3 else 0.0,
                             fx=float(y[4]) if len(y) > 5 else 0.0,   # 9/10 列标签第5,6位=Fx,Fy
                             fy=float(y[5]) if len(y) > 5 else 0.0,
                             mag=mag, hz=stats.hz(), contact=bool(contact), t=now)
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stop.set()
        srv.shutdown()


if __name__ == "__main__":
    main()
