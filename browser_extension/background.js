
chrome.runtime.onMessage.addListener(
  (message, sender, sendResponse) => {
    if (message.type !== "ANALYZE_REVIEW") {
      return;
    }

    fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: message.text
      })
    })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail || "Prediction failed."
          );
        }

        sendResponse({
          success: true,
          sentiment: data.sentiment,
          confidence: data.confidence
        });
      })
      .catch((error) => {
        sendResponse({
          success: false,
          error:
            "Cannot connect to the sentiment API. " +
            "Check that the API is running."
        });
      });

    // Keep the message channel open for the async response.
    return true;
  }
);
