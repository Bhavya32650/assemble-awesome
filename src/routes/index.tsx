import { Fragment, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { IRIS, CLASSES, CLASS_COLORS, type IrisClass } from "@/lib/iris-data";
import {
  computeScaler,
  scale,
  knnPredict,
  trainTestSplit,
  evaluate,
  type Vec4,
} from "@/lib/knn";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Iris KNN Classifier — DecodeLabs P2" },
      { name: "description", content: "Interactive K-Nearest Neighbors classifier for the Iris benchmark. Train, predict, and validate in your browser." },
      { property: "og:title", content: "Iris KNN Classifier — DecodeLabs P2" },
      { property: "og:description", content: "Supervised learning, live. Tune K, scale features, and see precision, recall, and F1." },
    ],
  }),
  component: IrisLab,
});

type Feature = "sl" | "sw" | "pl" | "pw";
const FEATURE_META: Record<Feature, { label: string; min: number; max: number; step: number; unit: string }> = {
  sl: { label: "Sepal length", min: 4.0, max: 8.0, step: 0.1, unit: "cm" },
  sw: { label: "Sepal width",  min: 2.0, max: 4.5, step: 0.1, unit: "cm" },
  pl: { label: "Petal length", min: 1.0, max: 7.0, step: 0.1, unit: "cm" },
  pw: { label: "Petal width",  min: 0.1, max: 2.6, step: 0.1, unit: "cm" },
};

function IrisLab() {
  const [features, setFeatures] = useState<Record<Feature, number>>({
    sl: 5.8, sw: 3.0, pl: 3.8, pw: 1.2,
  });
  const [k, setK] = useState(5);
  const [xAxis, setXAxis] = useState<Feature>("pl");
  const [yAxis, setYAxis] = useState<Feature>("pw");

  const { train, test } = useMemo(() => trainTestSplit(0.2, 42), []);
  const scaler = useMemo(() => computeScaler(train), [train]);
  const trainScaled = useMemo(
    () => train.map((s) => scale([s.sl, s.sw, s.pl, s.pw], scaler)),
    [train, scaler],
  );

  const query: Vec4 = [features.sl, features.sw, features.pl, features.pw];
  const queryScaled = scale(query, scaler);
  const { predicted, neighbors, votes } = useMemo(
    () => knnPredict(train, trainScaled, queryScaled, k),
    [train, trainScaled, queryScaled, k],
  );

  const evalResult = useMemo(() => evaluate(k, 0.2), [k]);

  return (
    <div className="min-h-screen px-4 py-10 md:px-10">
      <Header />

      <main className="mx-auto max-w-7xl space-y-8">
        <HeroBlock predicted={predicted} k={k} />

        <section className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <Card title="01 · Input" tag="Feature vector">
            <div className="space-y-5">
              {(Object.keys(FEATURE_META) as Feature[]).map((f) => (
                <SliderRow
                  key={f}
                  label={FEATURE_META[f].label}
                  value={features[f]}
                  min={FEATURE_META[f].min}
                  max={FEATURE_META[f].max}
                  step={FEATURE_META[f].step}
                  unit={FEATURE_META[f].unit}
                  onChange={(v) => setFeatures((p) => ({ ...p, [f]: v }))}
                />
              ))}

              <div className="border-t border-border pt-5">
                <SliderRow
                  label="K (neighbors)"
                  value={k}
                  min={1}
                  max={25}
                  step={1}
                  unit=""
                  onChange={(v) => setK(Math.round(v))}
                  accent
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  K=1 overfits noise. K too large smears boundaries. Find the elbow.
                </p>
              </div>
            </div>
          </Card>

          <Card title="02 · Process" tag="KNN · Euclidean in scaled space">
            <ScatterPlot
              train={train}
              xKey={xAxis}
              yKey={yAxis}
              query={query}
              neighbors={neighbors.map((n) => n.sample)}
              onXChange={setXAxis}
              onYChange={setYAxis}
            />
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-3">
          <Card title="03 · Output" tag="Prediction">
            <PredictionPanel predicted={predicted} votes={votes} k={k} />
          </Card>
          <Card title="04 · Metrics" tag={`Test set · n=${test.length}`}>
            <MetricsPanel evalResult={evalResult} />
          </Card>
          <Card title="05 · Confusion" tag="Actual × Predicted">
            <ConfusionMatrix confusion={evalResult.confusion} />
          </Card>
        </section>

        <Pipeline />

        <Footer />
      </main>
    </div>
  );
}

/* ───────── Layout pieces ───────── */

function Header() {
  return (
    <header className="mx-auto mb-10 flex max-w-7xl items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-lg bg-primary/15 ring-1 ring-primary/40">
          <span className="font-display text-lg font-bold text-primary">D</span>
        </div>
        <div>
          <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">DecodeLabs · Project 2</div>
          <div className="font-display text-sm font-semibold">Industrial Training Kit</div>
        </div>
      </div>
      <div className="hidden items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur md:flex">
        <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
        Live · scikit-learn parity in TypeScript
      </div>
    </header>
  );
}

function HeroBlock({ predicted, k }: { predicted: IrisClass; k: number }) {
  return (
    <section className="relative overflow-hidden rounded-3xl border border-border bg-card/60 px-6 py-10 backdrop-blur md:px-12 md:py-14">
      <div className="absolute inset-0 -z-10 opacity-70" style={{ background: "radial-gradient(60% 60% at 80% 20%, color-mix(in oklab, var(--color-primary) 20%, transparent), transparent 70%)" }} />
      <div className="text-xs uppercase tracking-[0.3em] text-primary">Data Classification · Supervised Learning</div>
      <h1 className="font-display mt-3 text-balance text-4xl font-bold leading-[1.05] md:text-6xl">
        From raw measurements to <span className="text-primary">intelligent decisions</span>.
      </h1>
      <p className="mt-4 max-w-2xl text-muted-foreground md:text-lg">
        Move four sliders. The model standardizes your input, finds the K closest historic flowers, and votes a species.
        Everything runs locally — no API, no mock, just the math.
      </p>
      <div className="mt-6 inline-flex items-center gap-3 rounded-full border border-border bg-background/60 px-4 py-2 text-sm">
        <span className="text-muted-foreground">Current prediction</span>
        <span
          className="font-mono font-semibold uppercase tracking-wider"
          style={{ color: CLASS_COLORS[predicted] }}
        >
          Iris {predicted}
        </span>
        <span className="text-muted-foreground">· K = {k}</span>
      </div>
    </section>
  );
}

function Card({ title, tag, children }: { title: string; tag?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-border bg-card/70 p-6 backdrop-blur transition-colors hover:border-primary/40">
      <div className="mb-5 flex items-baseline justify-between gap-3">
        <h2 className="font-display text-lg font-semibold">{title}</h2>
        {tag && <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{tag}</span>}
      </div>
      {children}
    </div>
  );
}

function SliderRow({ label, value, min, max, step, unit, onChange, accent }: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (v: number) => void; accent?: boolean;
}) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className={`font-mono text-sm font-semibold ${accent ? "text-primary" : ""}`}>
          {value.toFixed(step < 1 ? 1 : 0)}{unit && ` ${unit}`}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-primary"
      />
    </div>
  );
}

/* ───────── Scatter Plot (SVG) ───────── */

function ScatterPlot({ train, xKey, yKey, query, neighbors, onXChange, onYChange }: {
  train: { sl: number; sw: number; pl: number; pw: number; cls: IrisClass }[];
  xKey: Feature; yKey: Feature;
  query: Vec4;
  neighbors: { sl: number; sw: number; pl: number; pw: number; cls: IrisClass }[];
  onXChange: (f: Feature) => void;
  onYChange: (f: Feature) => void;
}) {
  const W = 560, H = 360, PAD = 36;
  const xs = train.map((s) => s[xKey]);
  const ys = train.map((s) => s[yKey]);
  const xMin = Math.min(...xs, query[featureIdx(xKey)]) - 0.2;
  const xMax = Math.max(...xs, query[featureIdx(xKey)]) + 0.2;
  const yMin = Math.min(...ys, query[featureIdx(yKey)]) - 0.2;
  const yMax = Math.max(...ys, query[featureIdx(yKey)]) + 0.2;
  const sx = (v: number) => PAD + ((v - xMin) / (xMax - xMin)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - yMin) / (yMax - yMin)) * (H - 2 * PAD);

  const neighborSet = new Set(neighbors);
  const qx = sx(query[featureIdx(xKey)]);
  const qy = sy(query[featureIdx(yKey)]);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">X:</span>
        <AxisPicker value={xKey} onChange={onXChange} />
        <span className="ml-3 text-muted-foreground">Y:</span>
        <AxisPicker value={yKey} onChange={onYChange} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-background/40 ring-1 ring-border">
        {/* grid */}
        {Array.from({ length: 6 }).map((_, i) => {
          const y = PAD + (i * (H - 2 * PAD)) / 5;
          const x = PAD + (i * (W - 2 * PAD)) / 5;
          return (
            <g key={i} stroke="currentColor" className="text-border" strokeDasharray="2 4">
              <line x1={PAD} y1={y} x2={W - PAD} y2={y} />
              <line x1={x} y1={PAD} x2={x} y2={H - PAD} />
            </g>
          );
        })}
        {/* training points */}
        {train.map((s, i) => {
          const isNeighbor = neighborSet.has(s);
          return (
            <circle
              key={i}
              cx={sx(s[xKey])}
              cy={sy(s[yKey])}
              r={isNeighbor ? 6 : 3.5}
              fill={CLASS_COLORS[s.cls]}
              fillOpacity={isNeighbor ? 1 : 0.55}
              stroke={isNeighbor ? "white" : "none"}
              strokeWidth={isNeighbor ? 1.5 : 0}
            />
          );
        })}
        {/* connector lines */}
        {neighbors.map((n, i) => (
          <line
            key={i}
            x1={qx} y1={qy}
            x2={sx(n[xKey])} y2={sy(n[yKey])}
            stroke={CLASS_COLORS[n.cls]}
            strokeOpacity={0.6}
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        ))}
        {/* query point */}
        <circle cx={qx} cy={qy} r={10} fill="none" stroke="white" strokeWidth={2} />
        <circle cx={qx} cy={qy} r={4} fill="white" />
        {/* axis labels */}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="currentColor" className="text-muted-foreground">{FEATURE_META[xKey].label} (cm)</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize="11" fill="currentColor" className="text-muted-foreground" transform={`rotate(-90 12 ${H / 2})`}>{FEATURE_META[yKey].label} (cm)</text>
      </svg>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs">
        {CLASSES.map((c) => (
          <span key={c} className="flex items-center gap-2 text-muted-foreground">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: CLASS_COLORS[c] }} /> {c}
          </span>
        ))}
        <span className="flex items-center gap-2 text-muted-foreground">
          <span className="h-3 w-3 rounded-full border-2 border-white" /> your sample
        </span>
      </div>
    </div>
  );
}

function AxisPicker({ value, onChange }: { value: Feature; onChange: (f: Feature) => void }) {
  return (
    <div className="inline-flex rounded-md border border-border bg-background/40 p-0.5">
      {(Object.keys(FEATURE_META) as Feature[]).map((f) => (
        <button
          key={f}
          onClick={() => onChange(f)}
          className={`rounded px-2 py-1 font-mono text-[10px] uppercase transition-colors ${
            value === f ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {f}
        </button>
      ))}
    </div>
  );
}

function featureIdx(f: Feature): number {
  return { sl: 0, sw: 1, pl: 2, pw: 3 }[f];
}

/* ───────── Output panels ───────── */

function PredictionPanel({ predicted, votes, k }: {
  predicted: IrisClass;
  votes: Record<IrisClass, number>;
  k: number;
}) {
  return (
    <div>
      <div
        className="rounded-xl border border-border p-5"
        style={{ background: `linear-gradient(135deg, ${CLASS_COLORS[predicted]}22, transparent 70%)`, borderColor: `${CLASS_COLORS[predicted]}66` }}
      >
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Predicted species</div>
        <div className="font-display mt-1 text-3xl font-bold" style={{ color: CLASS_COLORS[predicted] }}>
          Iris {predicted}
        </div>
      </div>
      <div className="mt-4 space-y-2">
        {CLASSES.map((c) => {
          const pct = (votes[c] / k) * 100;
          return (
            <div key={c}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="capitalize text-muted-foreground">{c}</span>
                <span className="font-mono">{votes[c]}/{k}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-background/60">
                <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: CLASS_COLORS[c] }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricsPanel({ evalResult }: { evalResult: ReturnType<typeof evaluate> }) {
  return (
    <div>
      <div className="grid grid-cols-2 gap-3">
        <Stat label="Accuracy" value={`${(evalResult.accuracy * 100).toFixed(1)}%`} />
        <Stat label="Macro F1" value={evalResult.macroF1.toFixed(3)} accent />
      </div>
      <div className="mt-4 space-y-2 text-xs">
        <div className="grid grid-cols-4 gap-2 font-mono uppercase text-muted-foreground">
          <span>class</span><span className="text-right">prec</span><span className="text-right">rec</span><span className="text-right">F1</span>
        </div>
        {evalResult.perClass.map((p) => (
          <div key={p.cls} className="grid grid-cols-4 gap-2 font-mono">
            <span style={{ color: CLASS_COLORS[p.cls] }}>{p.cls.slice(0, 4)}</span>
            <span className="text-right">{p.precision.toFixed(2)}</span>
            <span className="text-right">{p.recall.toFixed(2)}</span>
            <span className="text-right">{p.f1.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`font-display mt-1 text-2xl font-bold ${accent ? "text-primary" : ""}`}>{value}</div>
    </div>
  );
}

function ConfusionMatrix({ confusion }: { confusion: number[][] }) {
  const max = Math.max(...confusion.flat(), 1);
  return (
    <div>
      <div className="grid grid-cols-[auto_repeat(3,minmax(0,1fr))] gap-1.5 text-xs">
        <div />
        {CLASSES.map((c) => (
          <div key={c} className="text-center font-mono uppercase text-muted-foreground">{c.slice(0, 4)}</div>
        ))}
        {confusion.map((row, i) => (
          <Fragment key={`row${i}`}>
            <div className="flex items-center justify-end pr-1 font-mono uppercase" style={{ color: CLASS_COLORS[CLASSES[i]] }}>
              {CLASSES[i].slice(0, 4)}
            </div>
            {row.map((v, j) => {
              const onDiag = i === j;
              const intensity = v / max;
              return (
                <div
                  key={`c${i}${j}`}
                  className={`grid aspect-square place-items-center rounded font-mono text-sm ring-1 ${
                    onDiag ? "ring-primary/50" : "ring-border"
                  }`}
                  style={{
                    background: onDiag
                      ? `oklch(0.78 0.16 195 / ${0.15 + intensity * 0.6})`
                      : v > 0
                      ? `oklch(0.65 0.22 25 / ${0.15 + intensity * 0.6})`
                      : "transparent",
                  }}
                >
                  {v}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="mt-3 text-xs text-muted-foreground">Rows = actual. Cols = predicted. Diagonal = correct.</p>
    </div>
  );
}

/* ───────── Pipeline & Footer ───────── */

function Pipeline() {
  const steps = [
    { n: "I", title: "Input", body: "Iris benchmark · 150 samples · 4 features · 3 balanced classes." },
    { n: "P", title: "Process", body: "Shuffle → 80/20 split → StandardScaler (μ=0, σ=1) → KNN vote." },
    { n: "O", title: "Output", body: "Predicted class + per-class precision, recall, F1 + confusion matrix." },
  ];
  return (
    <section className="rounded-2xl border border-border bg-card/40 p-6 backdrop-blur md:p-8">
      <div className="mb-6 flex items-baseline justify-between">
        <h2 className="font-display text-xl font-semibold">The IPO blueprint</h2>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Project 2 architecture</span>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {steps.map((s) => (
          <div key={s.n} className="relative overflow-hidden rounded-xl border border-border bg-background/40 p-5">
            <div className="font-display absolute -right-3 -top-4 text-7xl font-bold text-primary/10">{s.n}</div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-primary">Stage</div>
            <div className="font-display mt-1 text-lg font-semibold">{s.title}</div>
            <p className="mt-2 text-sm text-muted-foreground">{s.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border pt-6 text-center text-xs text-muted-foreground">
      Built for DecodeLabs Industrial Training Kit · Batch 2026 · Pure-TypeScript KNN, no backend.
    </footer>
  );
}
