export interface Signal {
  label: string;
  value: React.ReactNode;
  tone?: "neutral" | "good" | "warn" | "accent";
}

export function SignalRibbon({ signals, label = "Run signals" }: { signals: Signal[]; label?: string }) {
  return (
    <section className="signal-ribbon" aria-label={label}>
      {signals.map((signal) => (
        <div className={`signal-cell tone-${signal.tone ?? "neutral"}`} key={signal.label}>
          <span>{signal.label}</span>
          <strong>{signal.value}</strong>
        </div>
      ))}
    </section>
  );
}
