import { IRIS, CLASSES, type IrisClass, type IrisSample } from "./iris-data";

export type Vec4 = [number, number, number, number];

export interface ScalerStats {
  mean: Vec4;
  std: Vec4;
}

export function computeScaler(samples: IrisSample[]): ScalerStats {
  const n = samples.length;
  const mean: Vec4 = [0, 0, 0, 0];
  for (const s of samples) {
    mean[0] += s.sl; mean[1] += s.sw; mean[2] += s.pl; mean[3] += s.pw;
  }
  for (let i = 0; i < 4; i++) mean[i] /= n;
  const variance: Vec4 = [0, 0, 0, 0];
  for (const s of samples) {
    variance[0] += (s.sl - mean[0]) ** 2;
    variance[1] += (s.sw - mean[1]) ** 2;
    variance[2] += (s.pl - mean[2]) ** 2;
    variance[3] += (s.pw - mean[3]) ** 2;
  }
  const std: Vec4 = variance.map((v) => Math.sqrt(v / n) || 1) as Vec4;
  return { mean, std };
}

export function scale(v: Vec4, s: ScalerStats): Vec4 {
  return [
    (v[0] - s.mean[0]) / s.std[0],
    (v[1] - s.mean[1]) / s.std[1],
    (v[2] - s.mean[2]) / s.std[2],
    (v[3] - s.mean[3]) / s.std[3],
  ];
}

function dist(a: Vec4, b: Vec4): number {
  let sum = 0;
  for (let i = 0; i < 4; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

export interface Neighbor {
  sample: IrisSample;
  distance: number;
  index: number;
}

export function knnPredict(
  train: IrisSample[],
  trainScaled: Vec4[],
  queryScaled: Vec4,
  k: number,
): { predicted: IrisClass; neighbors: Neighbor[]; votes: Record<IrisClass, number> } {
  const dists: Neighbor[] = trainScaled.map((v, i) => ({
    sample: train[i],
    distance: dist(v, queryScaled),
    index: i,
  }));
  dists.sort((a, b) => a.distance - b.distance);
  const nearest = dists.slice(0, k);
  const votes: Record<IrisClass, number> = { setosa: 0, versicolor: 0, virginica: 0 };
  for (const n of nearest) votes[n.sample.cls]++;
  let best: IrisClass = "setosa";
  let bestCount = -1;
  for (const c of CLASSES) {
    if (votes[c] > bestCount) { best = c; bestCount = votes[c]; }
  }
  return { predicted: best, neighbors: nearest, votes };
}

// Deterministic shuffle (mulberry32)
function rng(seed: number) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function trainTestSplit(testRatio = 0.2, seed = 42) {
  const indices = IRIS.map((_, i) => i);
  const r = rng(seed);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const cut = Math.floor(IRIS.length * (1 - testRatio));
  const train = indices.slice(0, cut).map((i) => IRIS[i]);
  const test = indices.slice(cut).map((i) => IRIS[i]);
  return { train, test };
}

export interface EvalResult {
  accuracy: number;
  confusion: number[][]; // 3x3, rows = actual, cols = predicted
  perClass: { cls: IrisClass; precision: number; recall: number; f1: number }[];
  macroF1: number;
}

export function evaluate(k: number, testRatio = 0.2): EvalResult {
  const { train, test } = trainTestSplit(testRatio);
  const scaler = computeScaler(train);
  const trainScaled = train.map((s) => scale([s.sl, s.sw, s.pl, s.pw], scaler));
  const confusion: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  let correct = 0;
  for (const s of test) {
    const q = scale([s.sl, s.sw, s.pl, s.pw], scaler);
    const { predicted } = knnPredict(train, trainScaled, q, k);
    const actualIdx = CLASSES.indexOf(s.cls);
    const predIdx = CLASSES.indexOf(predicted);
    confusion[actualIdx][predIdx]++;
    if (predicted === s.cls) correct++;
  }
  const perClass = CLASSES.map((cls, i) => {
    const tp = confusion[i][i];
    const fp = confusion.reduce((a, row, r) => a + (r !== i ? row[i] : 0), 0);
    const fn = confusion[i].reduce((a, v, c) => a + (c !== i ? v : 0), 0);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { cls, precision, recall, f1 };
  });
  const macroF1 = perClass.reduce((a, p) => a + p.f1, 0) / 3;
  return {
    accuracy: correct / test.length,
    confusion,
    perClass,
    macroF1,
  };
}
