const $ = (id) => document.getElementById(id);

// How far to trust the prediction, per animal group (your measured results)
const GROUP_NOTES = {
  scorpion: "On scorpion peptides alone, the model's balanced accuracy was about 0.64 in our tests (88 test peptides). Most scorpion examples are sodium or potassium channel toxins, so treat this as a preliminary result.",
  spider: "On spider peptides alone, the model's balanced accuracy was about 0.70 in our tests (42 test peptides). Few examples exist beyond enzymes and sodium channels, so treat this as a preliminary result.",
  cone_snail: "On cone snail peptides alone, the model's balanced accuracy was about 0.68 in our tests (68 test peptides). Some target classes have few examples, so treat this as a preliminary result.",
  unknown: "Without knowing the animal group, we can't say how reliable this result is."
};

function confidenceWords(p) {
  if (p >= 0.7) return "Strong signal";
  if (p >= 0.4) return "Possible, but not certain";
  return "Unclear";
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

function showResults(data, group) {
  $("target-name").textContent = data.target_prediction;
  $("target-words").textContent =
    confidenceWords(data.target_confidence) +
    " (" + Math.round(data.target_confidence * 100) + "% confidence)";

  const list = $("target-list");
  list.innerHTML = "";
  data.target_probabilities.forEach((row) => {
    const li = document.createElement("li");
    li.textContent = row.target_class + ": " + Math.round(row.probability * 100) + "%";
    list.appendChild(li);
  });

  $("group-note").textContent = GROUP_NOTES[group];

  if (data.toxicity_model_reliable) {
    $("tox-words").textContent =
      "Predicted: " + data.toxicity_prediction +
      " (" + Math.round(data.toxicity_confidence * 100) + "% confidence).";
    $("tox-card").classList.remove("muted");
  } else {
    $("tox-words").textContent =
      "Our toxicity model did not pass our reliability checks, so we do not show a toxicity verdict.";
    $("tox-card").classList.add("muted");
  }

  $("why-link").href = "/why?seq=" + encodeURIComponent(data.sequence);
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
}

$("go").addEventListener("click", async () => {
  $("error").hidden = true;
  $("results").hidden = true;
  $("go").disabled = true;
  $("go").textContent = "Analysing…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: $("seq").value })
    });
    const data = await res.json();
    if (!res.ok) { showError(data.error || "Something went wrong."); return; }
    showResults(data, $("group").value);
  } catch (e) {
    showError("Could not reach the server. Is app.py running?");
  } finally {
    $("go").disabled = false;
    $("go").textContent = "Analyse";
  }
});