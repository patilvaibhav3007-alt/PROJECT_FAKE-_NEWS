(() => {
  const $ = (sel) => document.querySelector(sel);
  const input = $("#newsInput");
  const predictBtn = $("#predictBtn");
  const clearBtn = $("#clearBtn");
  const resultCard = $("#resultCard");
  const badge = $("#predictionBadge");
  const confText = $("#confidenceText");
  const checkServer = $("#checkServer");
  const serverStatus = $("#serverStatus");

  async function callPredictAPI(text) {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  }

  function showResult(label, confidence) {
    const normalized = (label || "").toUpperCase();
    badge.classList.remove("true", "fake", "other");
    if (normalized === "TRUE") {
      badge.textContent = "TRUE NEWS";
      badge.classList.add("true");
    } else if (normalized === "FAKE") {
      badge.textContent = "FAKE NEWS";
      badge.classList.add("fake");
    } else if (normalized === "OTHER") {
      badge.textContent = "OTHER NEWS";
      badge.classList.add("other");
    } else {
      badge.textContent = normalized || "—";
    }
    confText.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%`;
    resultCard.classList.remove("hidden");
  }

  predictBtn?.addEventListener("click", async () => {
    const text = (input?.value || "").trim();
    if (!text) {
      alert("Please paste or type some text first.");
      return;
    }
    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing…";
    try {
      const data = await callPredictAPI(text);
      showResult(data.label, data.confidence ?? 0.0);
    } catch (err) {
      alert(err.message || String(err));
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Analyze";
    }
  });

  clearBtn?.addEventListener("click", () => {
    input.value = "";
    resultCard.classList.add("hidden");
    badge.classList.remove("true", "fake");
    badge.textContent = "—";
    confText.textContent = "Confidence: —";
  });

  checkServer?.addEventListener("click", async () => {
    serverStatus.textContent = "checking…";
    try {
      const res = await fetch("/health");
      const data = await res.json();
      serverStatus.textContent = data.status === "ok" ? "server ok" : "server issue";
    } catch {
      serverStatus.textContent = "server down";
    }
  });
})(); 

