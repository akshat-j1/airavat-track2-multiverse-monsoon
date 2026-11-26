(function (M) {
  'use strict';
  const count = 8;

  function sensitivityPerturbation(lastDay, random = Math.random) {
    const {height, width} = M.config;
    const channel = [0, 2, 3][Math.floor(random() * 3)];
    const centre = {r: Math.floor(random() * height), c: Math.floor(random() * width)};
    const radius = 3 + 3 * random();
    const magnitude = (random() < .5 ? -1 : 1) * (.25 + .55 * random());
    const delta = M.shockField(centre, radius, channel, magnitude);
    const [low, high] = channel === 0 ? [150, 350] : [-100, 100];
    delta[channel] = M.matrix((r, c) => {
      const value = lastDay[channel][r][c];
      return Math.min(high - value, Math.max(low - value, delta[channel][r][c]));
    });
    return delta;
  }

  function spreadBand(values) {
    const ordered = values.sort((a, b) => a - b);
    const middle = Math.floor(ordered.length / 2);
    return {low: ordered[0], high: ordered.at(-1),
      median: (ordered[middle] + ordered[Math.floor((ordered.length - 1) / 2)]) / 2};
  }

  function summarizeEnsemble(members, mask, thresholds) {
    if (!members.length || !members[0].length) throw new Error('An ensemble requires a nonempty forecast.');
    const {height, width} = M.config;
    const shape = [members[0].length, 6, height, width];
    if (!members.every(member => M.shape(member, shape))) throw new Error('Ensemble members must share a finite forecast shape.');
    if (!M.shape(mask, [height, width])) throw new Error('The ensemble requires a land mask.');
    const land = [];
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        if (mask[r][c] >= .5) land.push([r, c]);
      }
    }
    if (!land.length) throw new Error('The ensemble land mask has no land cells.');
    const samples = members.length * land.length;
    return members[0].map((_, day) => {
      let heat = 0, heavy = 0;
      const temperatures = [], rainfall = [];
      for (const rollout of members) {
        const [t2m, tp] = rollout[day];
        let temperatureTotal = 0, rainTotal = 0;
        for (const [r, c] of land) {
          const kelvin = t2m[r][c], rain = tp[r][c];
          temperatureTotal += kelvin;
          rainTotal += rain;
          if (kelvin > thresholds.heatC + 273.15) heat++;
          if (rain >= thresholds.heavyRainMm) heavy++;
        }
        temperatures.push(temperatureTotal / land.length - 273.15);
        rainfall.push(rainTotal / land.length);
      }
      return {temperature: spreadBand(temperatures), rain: spreadBand(rainfall),
        heatProbability: heat / samples, heavyProbability: heavy / samples};
    });
  }

  Object.assign(M, {ensemblePerturbedMembers: count, sensitivityPerturbation, summarizeEnsemble});
})(window.Monsoon);
