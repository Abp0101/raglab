"use client";

import { useEffect, useState } from "react";

import { Icon } from "@/components/icon";
import { PageIntro } from "@/components/page-intro";
import { StateNote } from "@/components/state-note";
import { apiFetch, DEMO_MODE, errorMessage } from "@/lib/api";
import { demoCollections, demoDocuments, demoJobs } from "@/lib/demo-data";
import { formatDate, shortId, titleCase } from "@/lib/format";
import type { Collection, CursorPage, DocumentRecord, IngestionJob } from "@/types/api";

export default function LibraryPage() {
  const [collections, setCollections] = useState<Collection[]>(demoCollections);
  const [selectedId, setSelectedId] = useState(demoCollections[0].collection_id);
  const [documents, setDocuments] = useState<DocumentRecord[]>(demoDocuments);
  const [jobs, setJobs] = useState<IngestionJob[]>(demoJobs);
  const [preview, setPreview] = useState(DEMO_MODE);
  const [panel, setPanel] = useState<"collection" | "upload" | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const selected = collections.find((collection) => collection.collection_id === selectedId) ?? collections[0];

  const loadCollection = async (collectionId: string) => {
    if (DEMO_MODE) return;
    try {
      const [documentPage, jobPage] = await Promise.all([
        apiFetch<CursorPage<DocumentRecord>>(`/collections/${collectionId}/documents?limit=100`),
        apiFetch<CursorPage<IngestionJob>>(`/collections/${collectionId}/ingestion-jobs?limit=100`),
      ]);
      setDocuments(documentPage.items);
      setJobs(jobPage.items);
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  useEffect(() => {
    if (DEMO_MODE) return;
    apiFetch<CursorPage<Collection>>("/collections?limit=100")
      .then((page) => {
        setCollections(page.items);
        if (page.items[0]) {
          setSelectedId(page.items[0].collection_id);
          return loadCollection(page.items[0].collection_id);
        }
      })
      .catch(() => setPreview(true));
  }, []);

  const chooseCollection = (collectionId: string) => {
    setSelectedId(collectionId);
    setNotice(null);
    void loadCollection(collectionId);
  };

  const createCollection = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (DEMO_MODE || preview) return setNotice("Connect the local API to create a collection.");
    const form = new FormData(event.currentTarget);
    try {
      const created = await apiFetch<Collection>("/collections", {
        method: "POST",
        body: JSON.stringify({ name: form.get("name"), description: form.get("description") || null }),
      });
      setCollections((current) => [created, ...current]);
      setSelectedId(created.collection_id);
      setDocuments([]);
      setJobs([]);
      setPanel(null);
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  const uploadDocument = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (DEMO_MODE || preview) return setNotice("Connect the local API to enqueue a document.");
    const form = new FormData(event.currentTarget);
    try {
      const job = await apiFetch<IngestionJob>(`/collections/${selectedId}/ingestion-jobs`, { method: "POST", body: form });
      setJobs((current) => [job, ...current]);
      setPanel(null);
      setNotice(`${job.file_name} was accepted as job ${shortId(job.job_id)}.`);
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  const deleteDocument = async (document: DocumentRecord) => {
    if (DEMO_MODE || preview) return setNotice("Connect the local API to delete documents.");
    if (!window.confirm(`Delete “${document.display_title}” from every shared index?`)) return;
    try {
      await apiFetch(`/documents/${document.document_id}`, { method: "DELETE" });
      setDocuments((current) => current.filter((item) => item.document_id !== document.document_id));
    } catch (error) {
      setNotice(errorMessage(error));
    }
  };

  return (
    <div className="page-stack library-page">
      <PageIntro index="03" kicker="CORPUS / PROVENANCE / LIFECYCLE" accessibleTitle="Evidence library" title={<>Evidence<br /><em>library.</em></>} description="Curate the shared document plane used by every framework, with durable ingestion state and source provenance kept visible." aside={<div className="button-cluster"><button className="button secondary" onClick={() => setPanel("collection")} type="button"><Icon name="plus" /> New collection</button><button className="button primary" onClick={() => setPanel("upload")} type="button"><Icon name="upload" /> Ingest PDF</button></div>} />

      {preview && <StateNote title="Preview ledger">Sample records are shown while the local API is offline. Mutation controls will not send requests.</StateNote>}
      {notice && <div className="notice-bar" role="status"><span>{notice}</span><button type="button" onClick={() => setNotice(null)} aria-label="Dismiss notice"><Icon name="close" /></button></div>}

      <div className="library-layout">
        <aside className="collection-index" aria-label="Collections">
          <div className="console-heading"><span className="eyebrow">COLLECTION INDEX</span><span>{collections.length.toString().padStart(2, "0")} TOTAL</span></div>
          {collections.map((collection, index) => (
            <button type="button" className="collection-tab" data-active={selectedId === collection.collection_id || undefined} onClick={() => chooseCollection(collection.collection_id)} key={collection.collection_id}>
              <span>0{index + 1}</span><div><strong>{collection.name}</strong><small>{collection.document_count} documents / updated {formatDate(collection.updated_at)}</small></div><Icon name="arrow" />
            </button>
          ))}
          <div className="collection-rule"><span>SHARED BY</span><strong>PY / LC / LG / LI / HS</strong></div>
        </aside>

        <section className="document-ledger">
          <div className="collection-header"><div><span className="eyebrow">ACTIVE COLLECTION / {shortId(selected?.collection_id ?? "")}</span><h2>{selected?.name ?? "No collection"}</h2><p>{selected?.description}</p></div><div className="collection-count"><strong>{documents.length.toString().padStart(2, "0")}</strong><span>visible records</span></div></div>
          <div className="ledger-head" aria-hidden="true"><span>STATE</span><span>DOCUMENT / PROVENANCE</span><span>PAGES</span><span>UPLOADED</span><span>ACTION</span></div>
          <div className="document-list">
            {documents.map((document) => (
              <article className="document-row" key={document.document_id}>
                <span className={`status-stamp status-${document.status}`}>{document.status}</span>
                <div className="document-name"><strong>{document.display_title}</strong><span>{document.file_name}</span><small>{document.authors.length ? document.authors.join(" / ") : "Author metadata unavailable"}</small></div>
                <span className="cell-mono">{document.page_count ?? "—"}</span>
                <span className="cell-mono">{formatDate(document.uploaded_at)}</span>
                <button className="row-action" type="button" onClick={() => void deleteDocument(document)} disabled={document.status === "processing" || document.status === "pending"}>Delete</button>
              </article>
            ))}
            {!documents.length && <div className="empty-ledger"><strong>No documents indexed</strong><p>Ingest a text-based PDF to create the first shared evidence record.</p></div>}
          </div>

          <section className="job-strip" aria-label="Recent ingestion jobs">
            <div className="section-heading"><div><span className="eyebrow">DURABLE QUEUE</span><h3>Recent ingestion state</h3></div><span className="array-tag">POSTGRES LEASES</span></div>
            {jobs.length ? jobs.slice(0, 4).map((job) => <div className="job-row" key={job.job_id}><span className={`job-pulse status-${job.status}`} /><strong>{job.file_name}</strong><span>{titleCase(job.status)}</span><span>attempt {job.attempt_count}</span><span>{shortId(job.job_id)}</span></div>) : <p className="quiet-copy">No queued or recent jobs for this collection.</p>}
          </section>
        </section>
      </div>

      {panel && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setPanel(null)}>
          <section className="settings-panel wide" role="dialog" aria-modal="true" aria-labelledby="library-panel-heading" onMouseDown={(event) => event.stopPropagation()}>
            <div className="panel-heading"><div><span className="eyebrow">LIBRARY MUTATION</span><h2 id="library-panel-heading">{panel === "collection" ? "Create collection" : "Queue document ingestion"}</h2></div><button className="icon-button" type="button" onClick={() => setPanel(null)} aria-label="Close"><Icon name="close" /></button></div>
            {panel === "collection" ? (
              <form onSubmit={createCollection}><label className="field-label" htmlFor="collection-name">Collection name</label><input className="text-input" id="collection-name" name="name" maxLength={255} required /><label className="field-label" htmlFor="collection-description">Research note</label><textarea className="query-textarea compact" id="collection-description" name="description" maxLength={4000} /><button className="button primary panel-submit" type="submit">Create shared collection <Icon name="arrow" /></button></form>
            ) : (
              <form onSubmit={uploadDocument}><p className="form-context">Target: <strong>{selected?.name}</strong></p><label className="file-drop" htmlFor="pdf-file"><Icon name="upload" /><strong>Select a text-based PDF</strong><span>Maximum 25 MB / validated before persistence</span><input id="pdf-file" name="file" type="file" accept="application/pdf,.pdf" required /></label><label className="field-label" htmlFor="display-title">Display title <span>optional</span></label><input className="text-input" id="display-title" name="display_title" maxLength={500} /><button className="button primary panel-submit" type="submit">Persist and enqueue <Icon name="arrow" /></button></form>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
