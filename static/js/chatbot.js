/**
 * chatbot.js
 * -----------
 * FEATURE 4 — AI Health Chatbot (frontend).
 *
 * Communicates with POST /api/chat (fallback /chatbot) with { message, history }
 * and renders conversation turns in the floating widget defined in templates/_chatbot.html.
 * Handles loading indicators, disabled input states during requests, and friendly error bubbles.
 */

(function () {
  const launcher = document.getElementById("chatbotLauncher");
  const win = document.getElementById("chatbotWindow");
  const closeBtn = document.getElementById("chatbotClose");
  const messagesEl = document.getElementById("chatbotMessages");
  const form = document.getElementById("chatbotForm");
  const input = document.getElementById("chatbotInput");
  const sendBtn = form ? form.querySelector(".chatbot-send") : null;

  if (!launcher || !win || !form || !input) return; // Not logged in / widget not present

  // In-memory session history
  let history = JSON.parse(sessionStorage.getItem("mediPredictChatHistory") || "[]");

  function persistHistory() {
    history = history.slice(-12);
    sessionStorage.setItem("mediPredictChatHistory", JSON.stringify(history));
  }

  function renderHistory() {
    history.forEach((turn) => {
      appendBubble(turn.text, turn.role === "user" ? "user" : "bot");
    });
  }

  function appendBubble(text, kind) {
    const bubble = document.createElement("div");
    bubble.className = `chatbot-bubble chatbot-bubble--${kind}`;
    bubble.textContent = text;
    messagesEl.appendChild(bubble);
    scrollToBottom();
    return bubble;
  }

  function showTyping() {
    const typing = document.createElement("div");
    typing.className = "chatbot-typing";
    typing.id = "chatbotTyping";
    typing.innerHTML = "<span>MediPredict AI is thinking...</span><div class='chatbot-typing-dots'><span></span><span></span><span></span></div>";
    messagesEl.appendChild(typing);
    scrollToBottom();
  }

  function hideTyping() {
    const typing = document.getElementById("chatbotTyping");
    if (typing) typing.remove();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function openChat() {
    win.classList.add("is-open");
    input.focus();
  }

  function closeChat() {
    win.classList.remove("is-open");
  }

  launcher.addEventListener("click", () => {
    win.classList.contains("is-open") ? closeChat() : openChat();
  });
  if (closeBtn) closeBtn.addEventListener("click", closeChat);

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 90) + "px";
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    if (message.length > 1000) {
      appendBubble("Please keep your question under 1000 characters.", "error");
      return;
    }

    appendBubble(message, "user");
    history.push({ role: "user", text: message });
    persistHistory();

    input.value = "";
    input.style.height = "auto";
    
    // Disable inputs during request (Loading State)
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    showTyping();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history: history.slice(0, -1) }),
      });

      const data = await response.json();
      hideTyping();

      if (!response.ok || data.success === false) {
        const errText = data.error || "The AI service is temporarily unavailable. Please try again later.";
        appendBubble(errText, "error");
      } else {
        const replyText = data.response || data.reply;
        if (replyText) {
          appendBubble(replyText, "bot");
          history.push({ role: "assistant", text: replyText });
          persistHistory();
        } else {
          appendBubble("Sorry, I didn't get a usable response. Please try rephrasing.", "error");
        }
      }
    } catch (err) {
      hideTyping();
      appendBubble("Unable to connect to the server. Please check your connection and try again.", "error");
    } finally {
      // Re-enable inputs
      input.disabled = false;
      if (sendBtn) sendBtn.disabled = false;
      input.focus();
    }
  });

  renderHistory();
})();
