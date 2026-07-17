import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import QueryPage from "@/app/query/page";

describe("Evidence workbench", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("exposes the framework controls and ranked evidence semantically", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline test")));
    render(<QueryPage />);

    expect(screen.getByRole("heading", { name: /evidence workbench/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Research question")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Select Langgraph framework" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("region", { name: "Grounded response" })).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "Retrieved evidence" })).toBeInTheDocument();
  });
});
