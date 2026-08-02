const createMedsam2 = require('./medsam2.js');
const fs = require('fs');
(async () => {
  const M = await createMedsam2();
  M.FS.mkdir('/medsam2');
  M.FS.writeFile('/medsam2/hiera_weights_fp16.bin', new Uint8Array(fs.readFileSync('medsam2/hiera_weights_fp16.bin')));
  M.FS.writeFile('/medsam2/weights_fp16.bin', new Uint8Array(fs.readFileSync('medsam2/dec_weights_fp16.bin')));
  M.FS.writeFile('/medsam2/config.txt', fs.readFileSync('medsam2/dec_config.txt','utf8'));
  console.log('fn_ready =', M.ccall('fn_ready','number',[],[]));
  const W=320,H=320; const rgba=new Uint8Array(W*H*4);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){ const i=(y*W+x)*4; const disc=((x-160)**2+(y-160)**2)<70*70?200:40; rgba[i]=rgba[i+1]=rgba[i+2]=disc; rgba[i+3]=255; }
  const p=M._malloc(rgba.length); M.HEAPU8.set(rgba,p);
  const t=Date.now(); M.ccall('fn_encode','number',['number','number','number'],[p,W,H]); M._free(p);
  console.log('encode:', ((Date.now()-t)/1000).toFixed(1),'s');
  const dp=M.ccall('fn_decode','number',['number','number'],[160,160]);
  let fg=0; for(let i=0;i<256*256;i++) if(M.HEAPF32[dp/4+i]>0) fg++;
  console.log('decode mask px>0 =', fg, '('+(100*fg/(256*256)).toFixed(1)+'%)  IoU', M.ccall('fn_iou','number',[],[]).toFixed(3));
  console.log(fg>0 ? 'WASM END-TO-END OK' : 'WASM ran but empty mask');
})();
