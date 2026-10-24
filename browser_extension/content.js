let popup = null;
let lastText = "";
let requestNumber = 0;
let selectionTimer = null;

function isExtensionValid() {
  return typeof chrome !== "undefined" && chrome.runtime && !!chrome.runtime.id;
}

function detectAspects(text) {
  const lower = text.toLowerCase();
  const aspects = [];
  if (lower.match(/\b(battery|charge|charging|drain|mah|life)\b/)) aspects.push("🔋 Battery");
  if (lower.match(/\b(camera|photo|picture|lens|zoom|megapixel|video|selfie)\b/)) aspects.push("📸 Camera");
  if (lower.match(/\b(screen|display|amoled|oled|resolution|brightness|hertz|hz)\b/)) aspects.push("📱 Display");
  if (lower.match(/\b(speed|lag|fast|processor|ram|snapdragon|bionic|game|gaming|performance)\b/)) aspects.push("⚡ Performance");
  if (lower.match(/\b(price|cheap|expensive|cost|money|value|worth|deal)\b/)) aspects.push("💰 Price/Value");
  if (lower.match(/\b(sound|speaker|audio|mic|microphone|volume|bass)\b/)) aspects.push("🎧 Audio");
  return aspects;
}

function showPopup(state, data = {}) {
  if (!popup) {
    popup = document.createElement("div");
    popup.id = "ai-review-sentiment-popup";

    popup.style.cssText = `
      position: fixed;
      right: 24px;
      bottom: 24px;
      z-index: 2147483647;
      width: 320px;
      padding: 18px;
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      box-shadow: 0 20px 35px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.15);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 13px;
      line-height: 1.5;
      backdrop-filter: blur(12px);
      box-sizing: border-box;
    `;

    document.documentElement.appendChild(popup);
  }

  popup.replaceChildren();

  // Header
  const header = document.createElement("div");
  header.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  `;

  const brand = document.createElement("div");
  brand.style.cssText = `
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #38bdf8;
  `;
  brand.innerHTML = `<span>✨</span><span>AI Review Inspector</span>`;

  const closeBtn = document.createElement("button");
  closeBtn.innerHTML = "&times;";
  closeBtn.style.cssText = `
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 20px;
    cursor: pointer;
    line-height: 1;
    padding: 0 4px;
  `;
  closeBtn.addEventListener("click", () => {
    popup?.remove();
    popup = null;
    lastText = "";
  });

  header.appendChild(brand);
  header.appendChild(closeBtn);
  popup.appendChild(header);

  if (state === "loading") {
    const box = document.createElement("div");
    box.style.cssText = "display: flex; flex-direction: column; gap: 8px;";
    box.innerHTML = `
      <div style="font-weight:600; color:#e2e8f0;">⏳ Running BERT Inference...</div>
      <div style="color:#94a3b8; font-style:italic; font-size:12px; background:rgba(255,255,255,0.04); padding:6px 10px; border-radius:6px; border-left:3px solid #38bdf8;">"${data.snippet || ""}"</div>
    `;
    popup.appendChild(box);

  } else if (state === "result") {
    const sentiment = (data.sentiment || "").toLowerCase();
    let themeColor = "#f59e0b";
    let emoji = "😐";

    if (sentiment === "positive") {
      themeColor = "#10b981";
      emoji = "😊";
    } else if (sentiment === "negative") {
      themeColor = "#ef4444";
      emoji = "😞";
    }

    const card = document.createElement("div");
    card.style.cssText = `
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid ${themeColor}50;
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 10px;
    `;

    const row = document.createElement("div");
    row.style.cssText = "display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;";

    const badge = document.createElement("span");
    badge.style.cssText = `font-weight:800; font-size:16px; color:${themeColor}; text-transform:capitalize;`;
    badge.innerHTML = `${emoji} ${sentiment}`;

    const conf = document.createElement("span");
    conf.style.cssText = "font-weight:700; font-size:14px; color:#ffffff;";
    conf.textContent = (data.confidence * 100).toFixed(1) + "%";

    row.appendChild(badge);
    row.appendChild(conf);
    card.appendChild(row);

    // Progress bar
    const barTrack = document.createElement("div");
    barTrack.style.cssText = "background: rgba(255, 255, 255, 0.1); height: 5px; border-radius: 3px; overflow: hidden;";
    const barFill = document.createElement("div");
    barFill.style.cssText = `background: ${themeColor}; width: ${(data.confidence * 100).toFixed(1)}%; height: 100%; border-radius: 3px;`;
    barTrack.appendChild(barFill);
    card.appendChild(barTrack);

    popup.appendChild(card);

    // Aspects
    const aspects = detectAspects(data.rawText || "");
    if (aspects.length > 0) {
      const aspectRow = document.createElement("div");
      aspectRow.style.cssText = "display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;";
      aspects.forEach((asp) => {
        const chip = document.createElement("span");
        chip.style.cssText = `
          background: rgba(255, 255, 255, 0.08);
          color: #cbd5e1;
          font-size: 11px;
          padding: 3px 8px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        `;
        chip.textContent = asp;
        aspectRow.appendChild(chip);
      });
      popup.appendChild(aspectRow);
    }

  } else if (state === "error") {
    const errorBox = document.createElement("div");
    errorBox.style.cssText = `
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 12px;
    `;
    errorBox.innerHTML = `<strong>⚠️ ${data.title || "Notice"}</strong><br><span style="color:#e2e8f0; font-size:11px;">${data.detail || ""}</span>`;
    popup.appendChild(errorBox);
  }
}

function analyzeSelectedReview() {
  const selection = window.getSelection();
  const text = selection?.toString().trim() || "";

  if (text.length < 3 || text === lastText) {
    return;
  }

  // Prevent re-triggering analysis on text selected inside the popup itself
  if (popup && selection?.anchorNode && popup.contains(selection.anchorNode)) {
    return;
  }

  // Check if extension context is valid
  if (!isExtensionValid()) {
    showPopup("error", {
      title: "Extension Reloaded",
      detail: "Please refresh this webpage (press F5 or Ctrl+R) to reconnect.",
    });
    return;
  }

  lastText = text;
  const currentRequest = ++requestNumber;

  const snippet = text.length > 100 ? text.slice(0, 97) + "..." : text;
  showPopup("loading", { snippet });

  try {
    chrome.runtime.sendMessage(
      {
        type: "ANALYZE_REVIEW",
        text: text,
      },
      (result) => {
        if (currentRequest !== requestNumber) {
          return;
        }

        if (chrome.runtime.lastError || !result) {
          showPopup("error", {
            title: "API Connection Failed",
            detail: "Make sure the Python backend is running (`python run.py`).",
          });
          return;
        }

        if (!result.success) {
          showPopup("error", {
            title: "Prediction Error",
            detail: result.error || "Could not analyze text.",
          });
          return;
        }

        showPopup("result", {
          sentiment: result.sentiment,
          confidence: result.confidence,
          rawText: text,
        });
      }
    );
  } catch (err) {
    showPopup("error", {
      title: "Page Refresh Needed",
      detail: "The extension was updated. Please refresh this page (F5) to re-enable it.",
    });
  }
}

document.addEventListener("mouseup", () => {
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(analyzeSelectedReview, 350);
});

document.addEventListener("keyup", () => {
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(analyzeSelectedReview, 350);
});
