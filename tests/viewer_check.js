// Runs the page's sampler under Node: node tests/viewer_check.js page.html
// Prints one line per mechanism; throws if an accepted pose breaks a limit or collision rule.
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const code = html.split('<script id="cspace-math">')[1].split("</script>")[0];
vm.runInThisContext(code + "\nglobalThis.CSPACE = CSPACE;");
const { SYSTEMS, globalSample, localSample, freedoms } = CSPACE;
for (const sys of SYSTEMS) {
  const p = sys.params.map(q => q.value);
  const geo = sys.geometry(p);
  let scale = 0;   // same rule as the page's sceneScale()
  for (const b of geo) {
    for (const f of b.frames) scale = Math.max(scale, ...f.map(Math.abs));
    for (const pr of b.prims) {
      const t = [pr.m[3], pr.m[7], pr.m[11]].map(Math.abs);
      const ext = pr.type === "box" ? Math.max(...pr.size.map(Math.abs)) : pr.type === "mesh" ? Math.max(...pr.tris.map(Math.abs)) : Math.abs(pr.r) + Math.abs(pr.h || 0);
      scale = Math.max(scale, ...t.map(v => v + ext));
    }
  }
  scale = Math.max(scale, 1);
  const t0 = Date.now();
  let g = 0, gTries = 0, l = 0, worst = 0, dims = new Set(), x = null, maxStep = 0;
  for (let i = 0; i < 15; i++) {
    const r = globalSample(sys, p, scale, 4000);
    gTries += r.tries || 0;
    if (r.ok) { g++; x = r.x; worst = Math.max(worst, r.maxF); dims.add(freedoms(sys, r.x, p)); }
  }
  if (x) for (let i = 0; i < 15; i++) {
    const r = localSample(sys, x, p, 0.08, scale);
    if (r.ok) { l++; maxStep = Math.max(maxStep, r.dist); x = r.x; worst = Math.max(worst, r.maxF);
      const G = sys.G(x, p); if (G.some(v => v < -1e-9) || !sys.OK(x, p)) throw new Error(sys.name + ": accepted an invalid pose"); }
  }
  const poses = x ? Object.values(sys.poses(x, p)).flat() : [];
  console.log(`${sys.name.padEnd(15)} global ${g}/15 (avg ${(gTries / 15).toFixed(1)} tries)  local ${l}/15  max residual ${worst.toExponential(1)}  freedoms ${[...dims].join(",")}  max local step ${maxStep.toFixed(3)}  poses finite ${poses.every(Number.isFinite)}  ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
