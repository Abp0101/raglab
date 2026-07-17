import { Icon } from "@/components/icon";

export function StateNote({
  title,
  children,
  tone = "neutral",
}: {
  title: string;
  children: React.ReactNode;
  tone?: "neutral" | "warning" | "good";
}) {
  return (
    <div className={`state-note state-${tone}`} role={tone === "warning" ? "alert" : "status"}>
      <Icon name={tone === "warning" ? "warning" : "check"} />
      <div><strong>{title}</strong><p>{children}</p></div>
    </div>
  );
}
