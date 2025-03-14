// js/app.js

// Chat elements
const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const messagesDiv = document.getElementById("messages");

// Connection elements
const connectBtn = document.getElementById("connectBtn");
const connectionStatus = document.getElementById("connectionStatus");

// Event listeners for chat
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});

// Event listener for Connect button
connectBtn.addEventListener("click", function () {
    // Show "Connecting..." message
    connectionStatus.style.display = "block";
    connectionStatus.textContent = "Connecting...";
    connectionStatus.className = "connection-status"; // Reset classes

    // Simulate connection delay (1 second)
    setTimeout(() => {
        // Simulate connection result (80% chance success)
        const success = Math.random() < 0.8;
        if (success) {
            connectionStatus.textContent = "Database is connected!";
            connectionStatus.classList.add("success");
        } else {
            connectionStatus.textContent = "Database is not connected.";
            connectionStatus.classList.add("error");
        }
    }, 1000);
});


function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    let source;
    if (window.location.href.endsWith("sql")) {
        source = 0;
    } else {
        source = 1;
    }

    let body = JSON.stringify({ query: text, source: source });
    console.log("Sending request with body:");
    console.log(body);

    // Clear input and disable it while "thinking"
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;
    sendBtn.textContent = "Thinking...";

    // You can update the UI with the response here
    // Append user message
    const userMsg = document.createElement("div");
    userMsg.className = "message-card message-user";
    userMsg.innerHTML = `<span class="message-icon icon-user">👤</span><span>${text}</span>`;
    messagesDiv.appendChild(userMsg);

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    fetch("https://geetu040-test-public.hf.space/query/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: body
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Received response:");
            console.log(data.response);

            const botMsg = document.createElement("div");
            botMsg.className = "message-card message-bot";
            botMsg.innerHTML = `<span class="message-icon icon-bot">🤖</span><span>${data.response}</span>`;
            messagesDiv.appendChild(botMsg);

            // Scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Re-enable input and button, set focus back
            userInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.textContent = "Send";
            userInput.focus();
        })
        .catch(error => console.error("Error sending query:", error));

}
