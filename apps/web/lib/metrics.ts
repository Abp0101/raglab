export interface MetricSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

export function parsePrometheus(input: string): MetricSample[] {
  return input
    .split("\n")
    .filter((line) => line && !line.startsWith("#"))
    .flatMap((line) => {
      const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(-?\d+(?:\.\d+)?)$/);
      if (!match) return [];
      const labels: Record<string, string> = {};
      for (const pair of match[2]?.match(/\w+="(?:\\.|[^"])*"/g) ?? []) {
        const separator = pair.indexOf("=");
        labels[pair.slice(0, separator)] = pair.slice(separator + 2, -1);
      }
      return [{ name: match[1], labels, value: Number(match[3]) }];
    });
}

export function sumMetric(samples: MetricSample[], name: string): number {
  return samples.filter((sample) => sample.name === name).reduce((sum, sample) => sum + sample.value, 0);
}
