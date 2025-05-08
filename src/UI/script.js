const slider = document.getElementById("accuracy-slider");
const sliderValue = document.getElementById("slider-value");
const computeBtn = document.getElementById("compute-btn");
const outputContent = document.getElementById("output-content");

// Check if initialization is needed when page loads
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const response = await fetch("http://127.0.0.1:5000/status");
    const data = await response.json();
    
    if (data.status === "initializing") {
      outputContent.textContent = "System is initializing. Please wait...";
      startPollingStatus();
    }
  } catch (error) {
    console.error("Error checking initialization status:", error);
  }
});

slider.addEventListener("input", () => {
  sliderValue.textContent = `${slider.value}%`;
});

// Poll the status endpoint every few seconds when initialization is in progress
function startPollingStatus() {
  let pollingInterval = setInterval(async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/status");
      const data = await response.json();
      
      if (data.status === "ready") {
        outputContent.textContent = "System initialized and ready. Please select options and click Compute.";
        clearInterval(pollingInterval);
      } else {
        outputContent.textContent = data.message || "Still initializing...";
      }
    } catch (error) {
      console.error("Error polling status:", error);
    }
  }, 2000); // Poll every 2 seconds
}

computeBtn.addEventListener("click", async () => {
  const tradeoff = slider.value;
  const dataset = document.querySelector('input[name="dataset"]:checked')?.value;

  if (!dataset) {
    outputContent.textContent = "Please select a dataset.";
    return;
  }

  // Disable the button and show loading state
  computeBtn.disabled = true;
  outputContent.textContent = "Computing...";

  try {
    const response = await fetch("http://127.0.0.1:5000/compute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, tradeoff }),
    });

    const data = await response.json();
    
    if (data.status === "initializing") {
      outputContent.textContent = data.result;
      startPollingStatus();
    } else {
      outputContent.textContent = data.result;
    }
  } catch (error) {
    outputContent.textContent = "Error connecting to server.";
    console.error(error);
  } finally {
    // Re-enable the button
    computeBtn.disabled = false;
  }
});

// Function to trigger initialization explicitly if needed
async function initializeSystem() {
  try {
    outputContent.textContent = "Starting system initialization...";
    
    const response = await fetch("http://127.0.0.1:5000/initialize", {
      method: "POST"
    });
    
    const data = await response.json();
    outputContent.textContent = data.message;
    
    if (data.status === "started" || data.status === "in_progress") {
      startPollingStatus();
    }
  } catch (error) {
    outputContent.textContent = "Error during initialization.";
    console.error(error);
  }
}