// GLOBAL CONTEXT FROM ANALYSIS
let documentContext = "";
let chatModal = null;

// Floating chat button opens modal
document.addEventListener("DOMContentLoaded", () => {

    chatModal = new bootstrap.Modal(document.getElementById("chatModal"));

    document.getElementById("chatButton").addEventListener("click", () => {
        chatModal.show();
    });

    // Send message on button click
    document.getElementById("sendChat").addEventListener("click", sendChatMessage);

    // Press enter to send
    document.getElementById("chatInput").addEventListener("keypress", function(e) {
        if (e.key === "Enter") sendChatMessage();
    });

    // Main analysis button
    document.getElementById("analyzeBtn").addEventListener("click", analyzePolicies);
});


// -----------------------------------------------------
// ANALYZE POLICIES (UPLOAD PDF + USER DATA)
// -----------------------------------------------------
function analyzePolicies() {

    const insuranceType = document.getElementById("insurance_type").value;

    // Build user object dynamically
    let user = {};
    if (insuranceType === "Health") {
        user.age = parseInt(document.getElementById("age").value);
        user.dependents = parseInt(document.getElementById("dependents").value);
    } 
    else if (insuranceType === "Life") {
        user.age = parseInt(document.getElementById("life_age").value);
        user.annual_income = parseInt(document.getElementById("annual_income").value);
    } 
    else {
        user.value = parseInt(document.getElementById("asset_value").value);
    }

    const files = document.getElementById("pdfs").files;
    if (files.length === 0) {
        alert("Please upload at least one PDF.");
        return;
    }

    let formData = new FormData();
    formData.append("insurance_type", insuranceType);
    formData.append("user", JSON.stringify(user));

    for (let f of files) formData.append("pdfs[]", f);

    document.getElementById("analyzeBtn").innerText = "Processing...";
    document.getElementById("analyzeBtn").disabled = true;

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {

        // Save context for chatbot
        documentContext = data.context;

        // Show results box
        document.getElementById("results").style.display = "block";

        // Show chart
        const chartImg = document.getElementById("chartImg");
        chartImg.src = "data:image/png;base64," + data.chart;

        // Display parsed table
        let html = "<table class='table table-bordered mt-3'><thead><tr>";
        const keys = Object.keys(data.parsed[0]);

        keys.forEach(k => html += `<th>${k}</th>`);
        html += "</tr></thead><tbody>";

        data.parsed.forEach(p => {
            html += "<tr>";
            keys.forEach(k => {
                let val = p[k];
                if (Array.isArray(val)) val = val.join("<br>");
                html += `<td>${val ?? ""}</td>`;
            });
            html += "</tr>";
        });

        html += "</tbody></table>";
        document.getElementById("parsedData").innerHTML = html;

        // Enable PDF download
        const pdfData = data.pdf;
        document.getElementById("downloadPDF").onclick = () => {
            const a = document.createElement("a");
            a.href = "data:application/pdf;base64," + pdfData;
            a.download = "CredenceX_Report.pdf";
            a.click();
        };

    })
    .finally(() => {
        document.getElementById("analyzeBtn").innerText = "Analyze Policies";
        document.getElementById("analyzeBtn").disabled = false;
    });
}




// -----------------------------------------------------
// CHATBOT LOGIC
// -----------------------------------------------------
function sendChatMessage() {
    const input = document.getElementById("chatInput");
    const text = input.value.trim();
    if (!text) return;

    input.value = "";

    // Display user bubble
    addChatBubble(text, "user");

    let formData = new FormData();
    formData.append("message", text);
    formData.append("context", documentContext);

    fetch("/chat", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        addChatBubble(data.answer, "bot");
    });
}


// -----------------------------------------------------
// ADD CHAT BUBBLE TO CHAT AREA
// -----------------------------------------------------
function addChatBubble(message, sender) {
    const chatArea = document.getElementById("chatArea");

    const div = document.createElement("div");
    div.className = sender === "user" ? "chat-bubble-user" : "chat-bubble-bot";
    div.innerHTML = message;

    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
}
