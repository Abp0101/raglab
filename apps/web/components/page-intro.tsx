export function PageIntro({
  index,
  kicker,
  title,
  accessibleTitle,
  description,
  aside,
}: {
  index: string;
  kicker: string;
  title: React.ReactNode;
  accessibleTitle: string;
  description: string;
  aside?: React.ReactNode;
}) {
  return (
    <header className="page-intro">
      <div className="intro-index" aria-hidden="true">{index}</div>
      <div className="intro-copy">
        <span className="eyebrow">{kicker}</span>
        <h1 aria-label={accessibleTitle}>{title}</h1>
        <p>{description}</p>
      </div>
      {aside && <div className="intro-aside">{aside}</div>}
    </header>
  );
}
