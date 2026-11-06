const $ = (id) => document.getElementById(id);
const seq = new URLSearchParams(location.search).get("seq") || "";
$("seq").textContent = seq;

async function load() {
  if (!seq) {
    $("status").textContent = "No sequence given. Go back and analyse a peptide first.";
    return;
  }
  try {
    const res = await fetch("/explain?seq=" + encodeURIComponent(seq));
    const data = await res.json();
    if (!res.ok) { $("status").textContent = data.error || "Something went wrong."; return; }

    const max = Math.max(...data.features.map((f) => f.value)) || 1;
    const box = $("bars");
    box.innerHTML = "";

    data.features.forEach((f) => {
      const row = document.createElement("div");
      row.className = "bar-row";

      const label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = f.label;
      const raw = document.createElement("span");
      raw.className = "raw";
      raw.textContent = " (" + f.feature + ")";
      label.appendChild(raw);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = (f.value / max) * 100 + "%";
      track.appendChild(fill);

      row.appendChild(label);
      row.appendChild(track);
      box.appendChild(row);
    });
    $("status").hidden = true;
  } catch (e) {
    $("status").textContent = "Could not reach the server.";
  }
}
load();