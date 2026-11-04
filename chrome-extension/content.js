function extractComments() {
    const elements = document.querySelectorAll(
        "ytd-comment-thread-renderer #content-text"
    );

    const comments = [];

    elements.forEach(element => {
        const text = element.innerText.trim();

        if (text && !comments.includes(text)) {
            comments.push(text);
        }
    });

    return comments;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extractComments") {
        const comments = extractComments();

        console.log(`[Extension] ${comments.length} commentaires extraits`);

        sendResponse({ comments });
    }
});