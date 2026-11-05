const API_URL = "https://friendly-palm-tree-pj65gxgq9x4xcr6g4-7860.app.github.dev";
let allResults = [];
let chart = null;

const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const dashboard = document.getElementById("dashboard");
const resultsContainer = document.getElementById("results");

// Analyze comments
analyzeBtn.addEventListener("click", async () => {
    loading.classList.remove("hidden");
    errorBox.classList.add("hidden");
    dashboard.classList.add("hidden");
    analyzeBtn.disabled = true;

    try {
        const [tab] = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });

        if (!tab.url || !tab.url.includes("youtube.com/watch")) {
            throw new Error("Veuillez ouvrir une vidéo YouTube.");
        }

        let response;

        try {
            response = await chrome.tabs.sendMessage(tab.id, {
                action: "extractComments"
            });
        } catch {
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ["content.js"]
            });

            response = await chrome.tabs.sendMessage(tab.id, {
                action: "extractComments"
            });
        }

        const comments = response?.comments;

        if (!comments || comments.length === 0) {
            throw new Error("Aucun commentaire visible. Faites défiler la section commentaires.");
        }

        console.log(`[API] ${comments.length} commentaires récupérés`);

        allResults = [];

        const batchSize = 50;

        for (let i = 0; i < comments.length; i += batchSize) {
            const batch = comments.slice(i, i + batchSize);

            const apiResponse = await fetch(`${API_URL}/predict_batch`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ comments: batch })
            });

            if (!apiResponse.ok) {
                throw new Error(`Erreur API : ${apiResponse.status}`);
            }

            const data = await apiResponse.json();

            allResults.push(...data.predictions);

            console.log(`[API] ${allResults.length}/${comments.length} traités`);
        }

        displayResults();
        dashboard.classList.remove("hidden");

    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        console.error(error);

    } finally {
        loading.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});

// Display results
function displayResults() {
    const total = allResults.length;

    if (total === 0) return;

    const counts = {
        Positive: 0,
        Neutral: 0,
        Negative: 0
    };

    allResults.forEach(item => {
        counts[item.sentiment]++;
    });

    document.getElementById("totalComments").textContent = total;
    document.getElementById("positivePercent").textContent =
        ((counts.Positive / total) * 100).toFixed(1) + "%";
    document.getElementById("neutralPercent").textContent =
        ((counts.Neutral / total) * 100).toFixed(1) + "%";
    document.getElementById("negativePercent").textContent =
        ((counts.Negative / total) * 100).toFixed(1) + "%";

    updateChart(counts);
    displayFilteredResults();
}

// Chart
function updateChart(counts) {
    const ctx = document.getElementById("sentimentChart");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Positif", "Neutre", "Négatif"],
            datasets: [{
                data: [
                    counts.Positive,
                    counts.Neutral,
                    counts.Negative
                ],
                backgroundColor: ["#159957", "#e0ad26", "#d64545"]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

// Filter results
document.getElementById("filter").addEventListener("change", displayFilteredResults);

function displayFilteredResults() {
    const filter = document.getElementById("filter").value;

    const filtered = filter === "all"
        ? allResults
        : allResults.filter(item => item.sentiment === filter);

    resultsContainer.innerHTML = "";

    filtered.forEach(item => {
        const div = document.createElement("div");
        div.className = "comment";

        const badge = document.createElement("span");
        badge.className = `badge ${item.sentiment.toLowerCase()}`;
        badge.textContent = item.sentiment;

        const text = document.createElement("p");
        text.textContent = item.comment;

        const confidence = document.createElement("span");
        confidence.className = "confidence";
        confidence.textContent =
            item.confidence !== undefined
                ? `Confiance : ${(item.confidence * 100).toFixed(1)}%`
                : "Confiance : non disponible";

        div.append(badge, text, confidence);
        resultsContainer.appendChild(div);
    });
}

// Dark mode
document.getElementById("themeBtn").addEventListener("click", () => {
    document.body.classList.toggle("dark");

    const dark = document.body.classList.contains("dark");

    document.getElementById("themeBtn").textContent =
        dark ? "Mode clair" : "Mode sombre";

    chrome.storage.local.set({ darkMode: dark });
});

// Restore theme
chrome.storage.local.get("darkMode", data => {
    if (data.darkMode) {
        document.body.classList.add("dark");
        document.getElementById("themeBtn").textContent = "Mode clair";
    }
});

// Copy results
document.getElementById("copyBtn").addEventListener("click", async () => {
    const text = allResults.map(item =>
        `${item.comment} | ${item.sentiment} | ${
            item.confidence !== undefined
                ? (item.confidence * 100).toFixed(1) + "%"
                : "N/A"
        }`
    ).join("\n");

    await navigator.clipboard.writeText(text);
    alert("Résultats copiés !");
});

// Export CSV
document.getElementById("exportBtn").addEventListener("click", () => {
    const rows = [
        ["Comment", "Sentiment", "Confidence"],
        ...allResults.map(item => [
            item.comment,
            item.sentiment,
            item.confidence ?? ""
        ])
    ];

    const csv = rows.map(row =>
        row.map(value =>
            `"${String(value).replace(/"/g, '""')}"`
        ).join(",")
    ).join("\n");

    const blob = new Blob(["\uFEFF" + csv], {
        type: "text/csv;charset=utf-8;"
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "youtube_sentiment_results.csv";
    link.click();

    URL.revokeObjectURL(url);
});