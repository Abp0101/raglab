import type { SVGProps } from "react";

export type IconName =
  | "overview"
  | "query"
  | "library"
  | "evaluation"
  | "operations"
  | "settings"
  | "arrow"
  | "plus"
  | "upload"
  | "close"
  | "check"
  | "warning";

export function Icon({ name, ...props }: SVGProps<SVGSVGElement> & { name: IconName }) {
  const paths: Record<IconName, React.ReactNode> = {
    overview: <><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4z"/><path d="M14 17h6M17 14v6"/></>,
    query: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5M8 10.5h5M10.5 8v5"/></>,
    library: <><path d="M4 5.5 8 4l4 1.5L16 4l4 1.5v14L16 18l-4 1.5L8 18l-4 1.5z"/><path d="M8 4v14M12 5.5v14M16 4v14"/></>,
    evaluation: <><path d="M4 20V9M10 20V4M16 20v-7M22 20H2"/><path d="m3 6 6-3 6 6 6-5"/></>,
    operations: <><path d="M3 12h4l2.2-6 4.1 12 2.2-6H21"/><circle cx="12" cy="12" r="10"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></>,
    arrow: <><path d="M5 12h14M14 7l5 5-5 5"/></>,
    plus: <path d="M12 5v14M5 12h14"/>,
    upload: <><path d="M12 16V3M7 8l5-5 5 5"/><path d="M4 14v6h16v-6"/></>,
    close: <path d="m6 6 12 12M18 6 6 18"/>,
    check: <path d="m5 12 4 4L19 6"/>,
    warning: <><path d="M12 3 2.5 20h19z"/><path d="M12 9v5M12 17.5v.5"/></>,
  };
  return (
    <svg aria-hidden="true" fill="none" height="24" viewBox="0 0 24 24" width="24" stroke="currentColor" strokeLinecap="square" strokeLinejoin="miter" strokeWidth="1.6" {...props}>
      {paths[name]}
    </svg>
  );
}
