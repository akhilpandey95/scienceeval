const fossilsMRoot = document.querySelector("[data-fossils-m-root]");
let fossilsMSelectsReady = false;

if (fossilsMRoot) {
  initFossilsM().catch((error) => {
    console.error("Failed to initialize Fossils-M", error);

    const status = document.querySelector("[data-fossils-m-status]");
    if (status) {
      status.textContent = "Unable to load the model fossil record.";
      status.hidden = false;
    }
  });
}

async function initFossilsM() {
  const catalog = await fetchFossilsMJSON("data/fossils-m-catalog.json");
  const params = new URLSearchParams(window.location.search);
  const modelId = resolveFossilsMModelId(catalog, params.get("model"));
  const model = catalog.models[modelId];

  renderFossilsM(catalog, modelId, model);
  initFossilsMSelects();
}

async function fetchFossilsMJSON(path) {
  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }

  return response.json();
}

function resolveFossilsMModelId(catalog, requestedId) {
  if (requestedId && catalog.models[requestedId]?.status === "available") {
    return requestedId;
  }

  return catalog.defaults.model;
}

function renderFossilsM(catalog, modelId, model) {
  const title = document.querySelector("title");
  const description = document.querySelector('meta[name="description"]');
  const status = document.querySelector("[data-fossils-m-status]");
  const modelValue = document.querySelector("[data-fossils-m-model-value]");
  const modelMenu = document.querySelector("[data-fossils-m-model-menu]");
  const plot = document.querySelector("[data-fossils-m-plot]");
  const headingLabel = document.querySelector("[data-fossils-m-heading-label]");
  const headingText = document.querySelector("[data-fossils-m-heading-text]");
  const missingNote = document.querySelector("[data-fossils-m-missing-note]");
  const modelTitle = document.querySelector("[data-fossils-m-title]");
  const coverage = document.querySelector("[data-fossils-m-coverage]");
  const modelType = document.querySelector("[data-fossils-m-model-type]");
  const readingRule = document.querySelector("[data-fossils-m-reading-rule]");
  const ledger = document.querySelector("[data-fossils-m-ledger]");
  const rows = document.querySelector("[data-fossils-m-rows]");

  document.title = model.page_title;

  if (description) {
    description.setAttribute("content", model.description);
  }

  if (title) {
    title.textContent = model.page_title;
  }

  if (modelValue) {
    modelValue.textContent = model.label;
  }

  if (plot) {
    plot.src = model.plot;
    plot.alt = model.plot_alt;
  }

  if (headingText) {
    headingText.textContent = `${model.heading_label}:`;
  } else if (headingLabel) {
    headingLabel.textContent = `${model.heading_label}:`;
  }

  if (missingNote) {
    const note = buildFossilsMMissingNote(model);
    missingNote.textContent = note;
    missingNote.hidden = !note;
  }

  if (modelTitle) {
    modelTitle.textContent = model.label;
  }

  if (coverage) {
    coverage.textContent = model.coverage;
  }

  if (modelType) {
    modelType.textContent = model.model_type;
  }

  if (readingRule) {
    readingRule.textContent = model.reading_rule;
  }

  if (ledger) {
    ledger.setAttribute("aria-label", model.ledger_aria_label);
  }

  if (status) {
    status.hidden = true;
  }

  renderFossilsMModelMenu(catalog, modelId, modelMenu);
  renderFossilsMRows(model.entries, rows);
}

function buildFossilsMMissingNote(model) {
  const missingEntries = model.entries.filter((entry) => entry.missing);
  if (!missingEntries.length) {
    return "";
  }

  const missingTitles = missingEntries.map((entry) => entry.title);
  if (missingTitles.length === 1) {
    return `missing ${missingTitles[0]}`;
  }
  if (missingTitles.length === 2) {
    return `missing ${missingTitles[0]} + ${missingTitles[1]}`;
  }

  return `missing ${missingTitles.length} fossils`;
}

function renderFossilsMModelMenu(catalog, currentModelId, menu) {
  if (!menu) {
    return;
  }

  menu.replaceChildren(
    ...Object.entries(catalog.models).map(([modelId, model]) => {
      const option = document.createElement("a");

      option.className = "catalog-select-option";
      if (modelId === currentModelId) {
        option.classList.add("is-current");
      }

      option.href = buildFossilsMUrl(catalog.defaults, modelId);
      option.textContent = model.label;
      return option;
    })
  );
}

function renderFossilsMRows(entries, container) {
  if (!container) {
    return;
  }

  container.innerHTML = entries
    .map((entry, index) => {
      const valueClass = entry.missing ? "model-sheet-entry-value is-missing" : "model-sheet-entry-value";
      const source = entry.source_url
        ? `<a href="${escapeFossilsMAttribute(entry.source_url)}" target="_blank" rel="noreferrer">${escapeFossilsMHTML(entry.source_label)}</a>`
        : `<span>${escapeFossilsMHTML(entry.source_label)}</span>`;

      return `
        <article class="model-sheet-entry" role="listitem">
          <div class="model-sheet-entry-head">
            <div>
              <p class="model-sheet-entry-index">${String(index + 1).padStart(2, "0")}</p>
              <h2 class="model-sheet-entry-title">
                <a href="${escapeFossilsMAttribute(entry.href)}">${escapeFossilsMHTML(entry.title)}<sup class="benchmark-origin-marker">${escapeFossilsMHTML(entry.origin)}</sup></a>
              </h2>
            </div>
            <p class="${valueClass}">${escapeFossilsMHTML(entry.value)}</p>
          </div>
          <p class="model-sheet-entry-metric">${escapeFossilsMHTML(entry.metric)}</p>
          <p class="model-sheet-entry-source">${source}</p>
        </article>
      `;
    })
    .join("");
}

function initFossilsMSelects() {
  if (!fossilsMRoot || fossilsMSelectsReady) {
    return;
  }

  const selects = [...fossilsMRoot.querySelectorAll(".catalog-select")];
  if (!selects.length) {
    return;
  }

  selects.forEach((select) => {
    const summary = select.querySelector("summary");
    const closeOtherSelects = () => {
      selects.forEach((otherSelect) => {
        if (otherSelect !== select) {
          otherSelect.open = false;
        }
      });
    };

    if (summary) {
      summary.addEventListener("pointerdown", () => {
        if (!select.open) {
          closeOtherSelects();
        }
      });

      summary.addEventListener("keydown", (event) => {
        if (!select.open && (event.key === "Enter" || event.key === " ")) {
          closeOtherSelects();
        }
      });
    }

    select.addEventListener("toggle", () => {
      if (select.open) {
        closeOtherSelects();
      }
    });
  });

  document.addEventListener("click", (event) => {
    if (!selects.some((select) => select.contains(event.target))) {
      selects.forEach((select) => {
        select.open = false;
      });
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      selects.forEach((select) => {
        select.open = false;
      });
    }
  });

  fossilsMSelectsReady = true;
}

function buildFossilsMUrl(defaults, modelId) {
  if (modelId === defaults.model) {
    return "fossils.html";
  }

  const params = new URLSearchParams({ model: modelId });
  return `fossils.html?${params.toString()}`;
}

function escapeFossilsMHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeFossilsMAttribute(value) {
  return escapeFossilsMHTML(value);
}
