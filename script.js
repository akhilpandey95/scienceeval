const reveals = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.18,
      rootMargin: "0px 0px -8% 0px",
    }
  );

  reveals.forEach((element) => observer.observe(element));
} else {
  reveals.forEach((element) => element.classList.add("visible"));
}

const benchmarkData = {
  frontierscience: {
    title: "FrontierScience",
    kind: "Research reasoning",
    signal: "Physics, chemistry, biology",
    summary:
      "FrontierScience is a hard science benchmark designed to measure expert-level reasoning rather than routine recall.",
    about:
      "It separates olympiad-style scientific problem solving from more open-ended research subtasks, which makes it useful when you want to see whether a frontier model can do more than answer familiar exam questions. It is especially relevant here because it asks what model weights can actually recover when the tasks start to resemble scientific work.",
    link: {
      href: "https://openai.com/index/frontierscience/",
      label: "Read the benchmark overview",
    },
  },
  gpqa: {
    title: "GPQA",
    kind: "Graduate QA",
    signal: "Difficult science questions",
    summary:
      "GPQA is a graduate-level science QA benchmark written to be hard to solve with shallow web lookup.",
    about:
      "Its value is that it still acts like a meaningful difficulty reference point for frontier models. If a model does well here, it suggests strong scientific recall and reasoning under pressure, but it still mostly captures question answering rather than broader research behavior.",
  },
  pubmedqa: {
    title: "PubMedQA",
    kind: "Biomedical QA",
    signal: "Abstract-grounded reading",
    summary:
      "PubMedQA evaluates whether a model can answer biomedical questions using the information in PubMed abstracts.",
    about:
      "This makes it a good check on evidence-sensitive reading: can the model stay close to the paper, track claims correctly, and return a defensible answer. It is useful when you care less about general cleverness and more about disciplined scientific reading.",
  },
  bioasq: {
    title: "BioASQ",
    kind: "Biomedical benchmark suite",
    signal: "Factoid, list, summary",
    summary:
      "BioASQ is a long-running biomedical benchmark suite spanning several answer formats instead of just one.",
    about:
      "That breadth matters because models can look strong on short factual responses while failing on list construction or evidence synthesis. BioASQ helps surface those differences and remains a useful benchmark when you want a broader picture of biomedical QA behavior.",
  },
  biored: {
    title: "BioRED",
    kind: "Relation extraction",
    signal: "Structured signal from papers",
    summary:
      "BioRED focuses on extracting biomedical entities and relations from scientific text.",
    about:
      "It is valuable because it tests whether a model can convert dense literature into structured representations rather than just produce fluent summaries. That makes it a strong benchmark for evaluating models that might support scientific databases, curation pipelines, or evidence extraction tools.",
  },
  scierc: {
    title: "SciERC",
    kind: "Scientific IE",
    signal: "Entities and relations",
    summary:
      "SciERC measures entity and relation extraction in scientific documents.",
    about:
      "Compared with QA benchmarks, SciERC is less about producing a final answer and more about preserving document structure. That makes it useful when evaluating whether a model can identify concepts, methods, tasks, and relationships in a way that could support downstream scientific tooling.",
  },
  mmlu: {
    title: "MMLU",
    kind: "General baseline",
    signal: "Broad capability reference",
    summary:
      "MMLU is a broad benchmark across many academic subjects, including science-heavy subdomains.",
    about:
      "It is not a specialized science benchmark, but it still matters as a baseline because it gives context for how much of a model's science performance is really domain-specific versus just part of a wider knowledge and reasoning profile.",
  },
  simpleqa: {
    title: "SimpleQA",
    kind: "Direct QA",
    signal: "Concise factual answers",
    summary:
      "SimpleQA gives a cleaner read on direct factual question answering without much surrounding scaffolding.",
    about:
      "That makes it useful as a contrast benchmark. If a model looks strong on concise factual QA but weak on science benchmarks with heavier reasoning or grounding requirements, the gap tells you something important about where the model's limits actually are.",
  },
  ei: {
    title: "EI",
    kind: "Open question",
    signal: "Capability slice still being framed",
    summary:
      "EI is currently a placeholder in the collection rather than a locked-in benchmark choice.",
    about:
      "It represents a useful gap in the index: there may still be important scientific capabilities not fully covered by the current set. Keeping it visible forces the collection to acknowledge that benchmark design is still an open curation problem, not a finished taxonomy.",
  },
};

const benchmarkButtons = document.querySelectorAll(".benchmark-card-button");
const benchmarkDetail = document.getElementById("benchmark-detail");
const benchmarkClose = document.getElementById("benchmark-detail-close");

if (benchmarkDetail && benchmarkButtons.length > 0) {
  const detailTitle = benchmarkDetail.querySelector(".benchmark-detail-title");
  const detailKind = benchmarkDetail.querySelector(".benchmark-detail-kind");
  const detailSignal = benchmarkDetail.querySelector(".benchmark-detail-signal");
  const detailSummary = benchmarkDetail.querySelector(".benchmark-detail-summary");
  const detailAbout = benchmarkDetail.querySelector(".benchmark-detail-about");
  const detailLink = benchmarkDetail.querySelector(".benchmark-detail-link");

  let activeBenchmarkId = null;

  const setActiveButtonState = (selectedId) => {
    benchmarkButtons.forEach((button) => {
      const isActive = button.dataset.benchmark === selectedId;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-expanded", String(isActive));
    });
  };

  const closeBenchmarkDetail = () => {
    activeBenchmarkId = null;
    benchmarkDetail.hidden = true;
    setActiveButtonState(null);
  };

  const openBenchmarkDetail = (benchmarkId) => {
    const benchmark = benchmarkData[benchmarkId];

    if (!benchmark) {
      return;
    }

    detailTitle.textContent = benchmark.title;
    detailKind.textContent = benchmark.kind;
    detailSignal.textContent = benchmark.signal;
    detailSummary.textContent = benchmark.summary;
    detailAbout.textContent = benchmark.about;

    if (benchmark.link) {
      detailLink.hidden = false;
      detailLink.href = benchmark.link.href;
      detailLink.textContent = benchmark.link.label;
    } else {
      detailLink.hidden = true;
      detailLink.removeAttribute("href");
      detailLink.textContent = "";
    }

    benchmarkDetail.hidden = false;
    activeBenchmarkId = benchmarkId;
    setActiveButtonState(benchmarkId);

    if (window.matchMedia("(max-width: 980px)").matches) {
      benchmarkDetail.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  benchmarkButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const benchmarkId = button.dataset.benchmark;

      if (benchmarkId === activeBenchmarkId) {
        closeBenchmarkDetail();
        return;
      }

      openBenchmarkDetail(benchmarkId);
    });
  });

  benchmarkClose?.addEventListener("click", closeBenchmarkDetail);
}
