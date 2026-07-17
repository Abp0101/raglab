import type { Metadata } from "next";
import "@fontsource-variable/instrument-sans";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./globals.css";

import { AppFrame } from "@/components/app-frame";

export const metadata: Metadata = {
  title: { default: "RAGLab Evidence Workbench", template: "%s / RAGLab" },
  description: "Inspect local retrieval evidence, framework behavior, and evaluation signals.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#workspace-content">
          Skip to workspace
        </a>
        <AppFrame>{children}</AppFrame>
      </body>
    </html>
  );
}
