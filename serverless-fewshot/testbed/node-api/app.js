// Node API function: plain http (no deps), realistic light runtime.
const T0 = Date.now();
const http = require("http");
const crypto = require("crypto");

// Simulate config/SDK init
let acc = "";
for (let i = 0; i < 2000; i++) {
  acc = crypto.createHash("sha256").update(acc + i).digest("hex");
}
const INIT_SEC = (Date.now() - T0) / 1000;

const port = process.env.PORT || 8080;
http.createServer((req, res) => {
  const t0 = Date.now();
  const h = crypto.createHash("sha256").update(String(Math.random())).digest("hex");
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    runtime: "node-api",
    init_sec: INIT_SEC,
    work_ms: Date.now() - t0,
    hash: h.slice(0, 8),
  }));
}).listen(port);
