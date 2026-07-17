"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { Icon, type IconName } from "@/components/icon";
import { API_KEY_STORAGE, API_URL, DEMO_MODE, storeApiKey } from "@/lib/api";

const navigation: { href: string; label: string; index: string; icon: IconName }[] = [
  { href: "/", label: "Overview", index: "01", icon: "overview" },
  { href: "/query", label: "Evidence", index: "02", icon: "query" },
  { href: "/library", label: "Library", index: "03", icon: "library" },
  { href: "/evaluation", label: "Evaluation", index: "04", icon: "evaluation" },
  { href: "/operations", label: "Operations", index: "05", icon: "operations" },
];

type ConnectionState = "checking" | "connected" | "offline" | "preview";

export function AppFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [connection, setConnection] = useState<ConnectionState>(DEMO_MODE ? "preview" : "checking");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const current = navigation.find((item) => item.href === pathname) ?? navigation[0];

  useEffect(() => {
    if (DEMO_MODE) return;
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`${API_URL}/health/live`, { cache: "no-store" });
        if (active) setConnection(response.ok ? "connected" : "offline");
      } catch {
        if (active) setConnection("offline");
      }
    };
    void check();
    const timer = window.setInterval(check, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const saveKey = () => {
    storeApiKey(apiKey);
    setSettingsOpen(false);
  };

  const openSettings = () => {
    setApiKey(sessionStorage.getItem(API_KEY_STORAGE) ?? "");
    setSettingsOpen(true);
  };

  return (
    <div className="app-frame">
      <aside className="side-rail" aria-label="Primary navigation">
        <Link className="brand" href="/" aria-label="RAGLab overview">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span className="brand-type">RAG<span>LAB</span></span>
          <span className="brand-edition">EVIDENCE / 01</span>
        </Link>
        <nav className="rail-nav">
          {navigation.map((item) => {
            const active = item.href === pathname;
            return (
              <Link className="rail-link" data-active={active || undefined} href={item.href} key={item.href} aria-current={active ? "page" : undefined}>
                <span className="rail-index">{item.index}</span>
                <Icon name={item.icon} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="rail-foot">
          <span>LOCAL FIRST</span>
          <span>BUILD 0.1.0</span>
        </div>
      </aside>

      <main className="main-stage">
        <header className="top-bar">
          <div className="route-title">
            <span className="eyebrow">WORKSPACE / {current.index}</span>
            <strong>{current.label}</strong>
          </div>
          <div className="top-actions">
            <span className={`connection-pill is-${connection}`} role="status">
              <i aria-hidden="true" />
              {connection === "checking" ? "Checking local API" : connection === "connected" ? "Local API online" : connection === "preview" ? "Preview dataset" : "API offline"}
            </span>
            <button className="icon-button" type="button" onClick={openSettings} aria-label="Connection settings">
              <Icon name="settings" />
            </button>
          </div>
        </header>
        <div className="workspace" id="workspace-content">{children}</div>
      </main>

      {settingsOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSettingsOpen(false)}>
          <section className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="connection-heading" onMouseDown={(event) => event.stopPropagation()}>
            <div className="panel-heading">
              <div><span className="eyebrow">LOCAL CONNECTION</span><h2 id="connection-heading">API access</h2></div>
              <button className="icon-button" type="button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><Icon name="close" /></button>
            </div>
            <p>Keys stay in this browser tab&apos;s session storage. They are never committed or sent anywhere except your configured RAGLab API.</p>
            <label className="field-label" htmlFor="api-url">API endpoint</label>
            <input className="text-input" id="api-url" value={API_URL} readOnly />
            <label className="field-label" htmlFor="api-key">Bearer key <span>optional locally</span></label>
            <input className="text-input" id="api-key" type="password" autoComplete="off" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Paste a configured API key" />
            <div className="panel-actions"><button className="button secondary" type="button" onClick={() => { setApiKey(""); storeApiKey(""); }}>Clear key</button><button className="button primary" type="button" onClick={saveKey}>Save for this tab <Icon name="arrow" /></button></div>
          </section>
        </div>
      )}
    </div>
  );
}
