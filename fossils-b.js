const fossilsBRoot = document.querySelector("[data-fossils-b-root]");

if (fossilsBRoot) {
  initFossilsB().catch((error) => {
    console.error("Failed to initialize Fossils-B", error);

    const status = document.querySelector("[data-fossils-b-status]");
    if (status) {
      status.textContent = "Unable to load the fossil record.";
    }
  });
}

async function initFossilsB() {
  const catalog = await fetchJSON("data/fossils-b-catalog.json");
  const params = new URLSearchParams(window.location.search);

  const specimenId = resolveSpecimenId(catalog, params.get("specimen"));
  const specimen = catalog.specimens[specimenId];
  const familyId = resolveFamilyId(specimen, params.get("family"), catalog.defaults.family);
  const family = specimen.families[familyId];
  const rows = sortRows(await fetchJSON(family.data));

  renderFossilsB(catalog, specimenId, specimen, familyId, family, rows);
}

async function fetchJSON(path) {
  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }

  return response.json();
}

function resolveSpecimenId(catalog, requestedId) {
  if (requestedId && catalog.specimens[requestedId]?.status === "available") {
    return requestedId;
  }

  return catalog.defaults.specimen;
}

function resolveFamilyId(specimen, requestedId, defaultFamilyId) {
  if (requestedId && specimen.families[requestedId]) {
    return requestedId;
  }

  if (specimen.families[defaultFamilyId]) {
    return defaultFamilyId;
  }

  return Object.keys(specimen.families)[0];
}

function sortRows(rows) {
  return [...rows].sort((left, right) => {
    const byDate = left.release_date.localeCompare(right.release_date);

    if (byDate !== 0) {
      return byDate;
    }

    return left.model.localeCompare(right.model);
  });
}

function renderFossilsB(catalog, specimenId, specimen, familyId, family, rows) {
  const title = document.querySelector("title");
  const description = document.querySelector('meta[name="description"]');
  const status = document.querySelector("[data-fossils-b-status]");
  const specimenValue = document.querySelector("[data-fossils-b-specimen-value]");
  const specimenMenu = document.querySelector("[data-fossils-b-specimen-menu]");
  const familyValue = document.querySelector("[data-fossils-b-family-value]");
  const familyMenu = document.querySelector("[data-fossils-b-family-menu]");
  const plot = document.querySelector("[data-fossils-b-plot]");
  const sectionLabel = document.querySelector("[data-fossils-b-section-label]");
  const heading = document.querySelector("[data-fossils-b-heading]");
  const copy = document.querySelector("[data-fossils-b-copy]");
  const ledger = document.querySelector("[data-fossils-b-ledger]");
  const ledgerRows = document.querySelector("[data-fossils-b-rows]");

  document.title = family.pageTitle;

  if (description) {
    description.setAttribute("content", family.description);
  }

  if (specimenValue) {
    specimenValue.textContent = specimen.label;
  }

  if (familyValue) {
    familyValue.textContent = family.label;
  }

  if (plot) {
    plot.src = family.plot;
    plot.alt = family.plotAlt;
  }

  if (sectionLabel) {
    sectionLabel.textContent = specimen.sectionLabel;
  }

  if (heading) {
    heading.textContent = specimen.sectionHeading;
  }

  if (copy) {
    copy.textContent = specimen.sectionCopy;
  }

  if (ledger) {
    ledger.setAttribute("aria-label", family.ledgerAriaLabel);
  }

  if (status) {
    status.hidden = true;
  }

  renderSpecimenMenu(catalog, specimenId, familyId, specimenMenu);
  renderFamilyMenu(catalog.defaults, specimen, specimenId, familyId, familyMenu);
  renderLedgerRows(rows, ledgerRows);
}

function renderSpecimenMenu(catalog, currentSpecimenId, currentFamilyId, menu) {
  if (!menu) {
    return;
  }

  menu.replaceChildren(
    ...Object.entries(catalog.specimens).map(([specimenId, specimen]) => {
      if (specimen.status !== "available") {
        const upcoming = document.createElement("span");
        const tag = document.createElement("span");

        upcoming.className = "catalog-select-option is-upcoming";
        upcoming.textContent = specimen.label;
        tag.textContent = "Soon";
        upcoming.appendChild(tag);
        return upcoming;
      }

      const option = document.createElement("a");

      option.className = "catalog-select-option";
      if (specimenId === currentSpecimenId) {
        option.classList.add("is-current");
      }

      option.href = buildFossilsBUrl(catalog.defaults, specimenId, resolveLinkedFamilyId(specimen, currentFamilyId, catalog.defaults.family));
      option.textContent = specimen.label;
      return option;
    })
  );
}

function renderFamilyMenu(defaults, specimen, specimenId, currentFamilyId, menu) {
  if (!menu) {
    return;
  }

  const items = [];

  Object.entries(specimen.families).forEach(([familyId, family]) => {
    const option = document.createElement("a");

    option.className = "catalog-select-option";
    if (familyId === currentFamilyId) {
      option.classList.add("is-current");
    }

    option.href = buildFossilsBUrl(defaults, specimenId, familyId);
    option.textContent = family.label;
    items.push(option);
  });

  (specimen.upcomingFamilies || []).forEach((family) => {
    const upcoming = document.createElement("span");
    const tag = document.createElement("span");

    upcoming.className = "catalog-select-option is-upcoming";
    upcoming.textContent = family.label;
    tag.textContent = "Soon";
    upcoming.appendChild(tag);
    items.push(upcoming);
  });

  menu.replaceChildren(...items);
}

function renderLedgerRows(rows, container) {
  if (!container) {
    return;
  }

  container.innerHTML = rows
    .map((row, index) => {
      const score = row.gpqa_diamond == null ? "&mdash;" : escapeHTML(Number(row.gpqa_diamond).toFixed(1));
      const scoreClass = row.gpqa_diamond == null ? "gpqa-ledger-score is-missing" : "gpqa-ledger-score";

      return `
        <article class="gpqa-ledger-row" role="row">
          <p class="gpqa-ledger-model" role="cell">
            <span class="gpqa-ledger-index">${String(index + 1).padStart(2, "0")}</span>
            <a href="${escapeAttribute(row.release_source_url)}" target="_blank" rel="noreferrer">${escapeHTML(row.model)}</a>
          </p>
          <p class="gpqa-ledger-release" role="cell">${escapeHTML(row.release_label)}</p>
          <p class="${scoreClass}" role="cell">${score}</p>
          <p class="gpqa-ledger-source" role="cell">
            <a href="${escapeAttribute(row.score_source_url)}" target="_blank" rel="noreferrer">${escapeHTML(row.score_source_label)}</a>
          </p>
          <p class="gpqa-ledger-note" role="cell">${escapeHTML(row.note)}</p>
        </article>
      `;
    })
    .join("");
}

function resolveLinkedFamilyId(specimen, requestedFamilyId, defaultFamilyId) {
  if (requestedFamilyId && specimen.families[requestedFamilyId]) {
    return requestedFamilyId;
  }

  if (specimen.families[defaultFamilyId]) {
    return defaultFamilyId;
  }

  return Object.keys(specimen.families)[0];
}

function buildFossilsBUrl(defaults, specimenId, familyId) {
  if (specimenId === defaults.specimen && familyId === defaults.family) {
    return "fossils-b.html";
  }

  const params = new URLSearchParams({
    specimen: specimenId,
    family: familyId,
  });

  return `fossils-b.html?${params.toString()}`;
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHTML(value);
}
