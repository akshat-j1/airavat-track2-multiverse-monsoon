(function (M) {
  'use strict';
  const {height: H, width: W} = M.config;
  const matrix = fn => Array.from({length: H}, (_, r) => Array.from({length: W}, (_, c) => fn(r, c)));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function shape(value, dimensions) {
    return dimensions.length ? Array.isArray(value) && value.length === dimensions[0]
      && value.every(item => shape(item, dimensions.slice(1))) : finite(value);
  }
  function normalizeGrid(data) {
    let {lat, lon} = data;
    if (shape(lat, [H]) && shape(lon, [W])) {
      const latAxis = lat, lonAxis = lon;
      lat = matrix(r => latAxis[r]); lon = matrix((r, c) => lonAxis[c]);
    }
    if (!shape(lat, [H, W]) || !shape(lon, [H, W])) throw new Error('Grid lat/lon must be finite 32/36 axes or 32×36 arrays.');
    if (lat.flat().some(x => Math.abs(x) > 90)) throw new Error('Grid latitude is outside −90…90.');
    const mask = data.land_sea_mask;
    if (mask !== undefined && !shape(mask, [H, W])) throw new Error('Land mask must be a finite 32×36 array.');
    if (data.orography_m !== undefined && !shape(data.orography_m, [H, W])) throw new Error('Orography must be a finite 32×36 array.');
    return {lat, lon, mask: mask || null, orography: data.orography_m || null};
  }
  function fallbackGrid() {
    return normalizeGrid({lat: Array.from({length: H}, (_, r) => 6.5 + r), lon: Array.from({length: W}, (_, c) => 65.5 + c)});
  }
  async function loadGrid() {
    try {
      const response = await fetch(new URL('grid.json', location.href));
      if (!response.ok) throw new Error(`grid.json HTTP ${response.status}`);
      return {...normalizeGrid(await response.json()), source: 'Dataset grid · coastline from land mask'};
    } catch (error) {
      return {...fallbackGrid(), source: 'Contract grid · 6.5–37.5°N, 65.5–100.5°E · no coastline', warning: error.message};
    }
  }
  const seasonal = day => [Math.sin(2 * Math.PI * day / 365), Math.cos(2 * Math.PI * day / 365)];
  function demoWindow(grid) {
    return Array.from({length: 3}, (_, day) => Array.from({length: 6}, (_, channel) => matrix((r, c) => {
      const lat = grid.lat[r][c], lon = grid.lon[r][c];
      const wet = Math.exp(-((lat - 18) ** 2 / 100 + (lon - 86) ** 2 / 80));
      const wave = Math.sin(lon / 6 + lat / 9 + day * .12);
      return [301 - .28 * (lat - 16) + 1.2 * wave,
        .8 + 7 * wet + .3 * (wave + 1),
        4 + 3 * Math.cos(lat / 9) + .7 * wave,
        1.5 + 2 * Math.sin(lon / 12) + .3 * day,
        101000 - 700 * wet + 90 * wave,
        34 + 18 * wet + 4 * wave][channel];
    })));
  }
  function parseWindow(value, rainUnit = 'mm') {
    let init = Array.isArray(value) ? value : value?.init_states;
    if (!shape(init, [3, 6, H, W])) throw new Error('Window must contain finite init_states with shape [3,6,32,36].');
    const doy = !Array.isArray(value) ? value.doy_sincos : undefined;
    if (doy !== undefined && !shape(doy, [2])) throw new Error('doy_sincos must be two finite numbers.');
    if (!['mm', 'm'].includes(rainUnit)) throw new Error('Uploaded rainfall must be in mm/day or m/day.');
    if (rainUnit === 'm') {
      init = init.map(day => day.map((field, channel) => channel === 1
        ? field.map(row => row.map(v => v * 1000)) : field));
      if (!shape(init, [3, 6, H, W])) throw new Error('Rainfall conversion produced a nonfinite input.');
    }
    return {init, doy};
  }
  function nearest(grid, lat, lon) {
    let best = Infinity, result = {r: 0, c: 0};
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const distance = (grid.lat[r][c] - lat) ** 2 + ((grid.lon[r][c] - lon) * Math.cos(lat * Math.PI / 180)) ** 2;
      if (distance < best) { best = distance; result = {r, c}; }
    }
    return result;
  }
  function shockField(centre, radius, channel, magnitude, secondary) {
    return Array.from({length: 6}, (_, k) => matrix((r, c) => {
      const d = Math.hypot(r - centre.r, c - centre.c);
      const weight = d < radius ? .5 * (1 + Math.cos(Math.PI * d / radius)) : 0;
      return weight * (k === channel ? magnitude : secondary && k === secondary.channel ? magnitude * secondary.ratio : 0);
    }));
  }
  async function api(path, payload) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch(M.config.api + path, {method: payload === undefined ? 'GET' : 'POST',
        headers: payload === undefined ? {} : {'Content-Type': 'application/json'},
        body: payload === undefined ? undefined : JSON.stringify(payload), signal: controller.signal});
      const body = await response.json();
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body)}`);
      if (path === '/simulate' && !shape(body.rollout, [payload.steps, 6, H, W])) throw new Error('Server rollout has an invalid shape or nonfinite values.');
      return body;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Request timed out after 60 seconds.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function difference(branch, prime) { return branch.map((channels, d) => channels.map((field, k) => field.map((row, r) => row.map((v, c) => v - prime[d][k][r][c])))); }
  function extent(rollouts, channel) {
    let min = Infinity, max = -Infinity;
    for (const rollout of rollouts) for (const day of rollout) for (const row of day[channel]) for (const v of row) { min = Math.min(min, v); max = Math.max(max, v); }
    if (min === max) { const pad = Math.max(.01, Math.abs(min) * .001); min -= pad; max += pad; }
    return [min, max];
  }
  function color(value, min, max, colors) {
    const position = Math.max(0, Math.min(1, (value - min) / (max - min))) * (colors.length - 1);
    const index = Math.min(colors.length - 2, Math.floor(position)), fraction = position - index;
    const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
    const a = rgb(colors[index]), b = rgb(colors[index + 1]);
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * fraction)).join(',')})`;
  }
  function canvasContext(canvas) {
    const {width, height} = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    const w = Math.max(1, width), h = Math.max(1, height);
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {ctx, w, h};
  }
  // Derive cell corners from coordinate centres, extrapolating at the outer edges.
  function corners(values) {
    const extended = (r, c) => {
      const rr = Math.max(0, Math.min(H - 1, r)), cc = Math.max(0, Math.min(W - 1, c));
      let v = values[rr][cc];
      if (r < 0) v += values[0][cc] - values[1][cc];
      if (r >= H) v += values[H - 1][cc] - values[H - 2][cc];
      if (c < 0) v += values[rr][0] - values[rr][1];
      if (c >= W) v += values[rr][W - 1] - values[rr][W - 2];
      return v;
    };
    return Array.from({length: H + 1}, (_, r) => Array.from({length: W + 1}, (_, c) =>
      (extended(r - 1, c - 1) + extended(r - 1, c) + extended(r, c - 1) + extended(r, c)) / 4));
  }
  function geometry(grid, w, h) {
    const lat = corners(grid.lat), lon = corners(grid.lon);
    const latValues = lat.flat(), lonValues = lon.flat();
    const south = Math.min(...latValues), north = Math.max(...latValues), west = Math.min(...lonValues), east = Math.max(...lonValues);
    const cos = Math.cos((north + south) * Math.PI / 360);
    const dx = (east - west) * cos, dy = north - south;
    if (dx <= 0 || dy <= 0) throw new Error('Grid coordinates have no spatial extent.');
    const scale = Math.min((w - 24) / dx, (h - 22) / dy);
    const left = (w - dx * scale) / 2, top = (h - dy * scale) / 2;
    const project = (latitude, longitude) => [left + (longitude - west) * cos * scale, top + (north - latitude) * scale];
    return {points: lat.map((row, r) => row.map((v, c) => project(v, lon[r][c]))), project};
  }
  function drawMap(canvas, grid, field, scale, colors, centre, radius, hover) {
    const {ctx, w, h} = canvasContext(canvas), geo = geometry(grid, w, h), p = geo.points;
    ctx.fillStyle = '#101d28'; ctx.fillRect(0, 0, w, h);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const polygon = [p[r][c], p[r][c + 1], p[r + 1][c + 1], p[r + 1][c]];
      ctx.beginPath(); polygon.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath();
      ctx.fillStyle = Number.isFinite(field[r][c]) ? color(field[r][c], ...scale, colors) : '#101d28'; ctx.fill();
      ctx.lineWidth = .25; ctx.strokeStyle = 'rgba(12,25,35,.12)'; ctx.stroke();
    }
    if (grid.mask) {
      ctx.beginPath();
      const line = (a, b) => {ctx.moveTo(...a); ctx.lineTo(...b);};
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const land = grid.mask[r][c] >= .5;
        if (c + 1 < W && land !== (grid.mask[r][c + 1] >= .5)) line(p[r][c + 1], p[r + 1][c + 1]);
        if (r + 1 < H && land !== (grid.mask[r + 1][c] >= .5)) line(p[r + 1][c], p[r + 1][c + 1]);
      }
      ctx.strokeStyle = 'rgba(12,23,31,.8)'; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,245,.85)'; ctx.lineWidth = 1; ctx.stroke();
    }
    // Radius is defined in cells, so trace its boundary through the geographic grid.
    ctx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const angle = i / 80 * 2 * Math.PI;
      const rr = Math.max(0, Math.min(H - 1, centre.r + radius * Math.sin(angle)));
      const cc = Math.max(0, Math.min(W - 1, centre.c + radius * Math.cos(angle)));
      const r0 = Math.floor(rr), c0 = Math.floor(cc), r1 = Math.min(H - 1, r0 + 1), c1 = Math.min(W - 1, c0 + 1);
      const interpolate = field => (field[r0][c0] * (1 - cc + c0) + field[r0][c1] * (cc - c0)) * (1 - rr + r0)
        + (field[r1][c0] * (1 - cc + c0) + field[r1][c1] * (cc - c0)) * (rr - r0);
      const point = geo.project(interpolate(grid.lat), interpolate(grid.lon));
      i ? ctx.lineTo(...point) : ctx.moveTo(...point);
    }
    ctx.strokeStyle = '#fff1c9'; ctx.lineWidth = 1.3; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    const [x, y] = geo.project(grid.lat[centre.r][centre.c], grid.lon[centre.r][centre.c]);
    ctx.strokeStyle = '#111d27'; ctx.lineWidth = 4; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#ffdc99'; ctx.fill();
    if (hover) {
      ctx.beginPath(); const {r, c} = hover;
      [p[r][c], p[r][c + 1], p[r + 1][c + 1], p[r + 1][c]].forEach((point, i) => i ? ctx.lineTo(...point) : ctx.moveTo(...point));
      ctx.closePath(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.7; ctx.stroke();
    }
    return geo;
  }
  function pickCell(canvas, grid, event) {
    const box = canvas.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
    const {points: p} = geometry(grid, box.width, box.height);
    // Point-in-polygon avoids selecting cells outside the geographic map.
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const polygon = [p[r][c], p[r][c + 1], p[r + 1][c + 1], p[r + 1][c]];
      let inside = false;
      for (let i = 0, j = 3; i < 4; j = i++) {
        const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (inside) return {r, c};
    }
    return null;
  }
  Object.assign(M, {matrix, shape, normalizeGrid, fallbackGrid, loadGrid, seasonal, demoWindow, parseWindow,
    nearest, shockField, api, difference, extent, color, canvasContext, drawMap, pickCell});
})(window.Monsoon);
