
let popup = null;
let lastText = "";
let requestNumber = 0;
let selectionTimer = null;

function showPopup(title, detail = "") {
  if (!popup) {
    popup = document.createElement("div");

    popup.style.cssText = `
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 2147483647;
      width: 290px;
      padding: 16px;
      background: #ffffff;
      color: #222222;
      border: 1px solid #dddddd;
      border-radius: 12px;
      box-shadow: 0 5px 25px #0003;
      font: 14px Arial, sans-serif;
    `;

    document.documentElement.appendChild(popup);
  }

  popup.replaceChildren();

  const heading = document.createElement("strong");
  heading.textContent = title;
  popup.appendChild(heading);

  if (detail) {
    const paragraph = document.createElement("p");
    paragraph.textContent = detail;
    paragraph.style.margin = "10px 0";
    popup.appendChild(paragraph);
  }

  const close = document.createElement("button");
  close.textContent = "Close";
  close.style.cssText = "padding: 6px 12px; cursor: pointer;";

  close.addEventListener("click", () => {
    popup?.remove();
    popup = null;
    lastText = "";
  });

  popup.appendChild(close);
}

function analyzeSelectedReview() {
  const text = window.getSelection()?.toString().trim() || "";

  if (text.length < 3 || text === lastText) {
    return;
  }

  lastText = text;
  const currentRequest = ++requestNumber;

  showPopup("Analyzing review...", text.slice(0, 160));

  chrome.runtime.sendMessage(
    {
      type: "ANALYZE_REVIEW",
      text: text
    },
    (result) => {
      if (currentRequest !== requestNumber) {
        return;
      }

      if (chrome.runtime.lastError || !result) {
        showPopup(
          "Connection error",
          "Reload the extension and make sure the Python API is running."
        );
        return;
      }

      if (!result.success) {
        showPopup("Prediction failed", result.error);
        return;
      }

      const confidence =
        (result.confidence * 100).toFixed(1) + "%";

      showPopup(
        "Sentiment: " + result.sentiment,
        "Model confidence: " + confidence
      );
    }
  );
}

document.addEventListener("mouseup", () => {
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(analyzeSelectedReview, 400);
});

document.addEventListener("keyup", () => {
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(analyzeSelectedReview, 400);
});
