import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

const LM_BASE = "https://dev.lunchmoney.app/v1";

// Add: global cancel flag (prevents ReferenceError in run/prompt)
let cancelled = false;

type LMCategory = {
  id: number;
  name: string;
  description?: string | null;
  archived?: boolean;
  is_group?: boolean;
  group_id?: number | null;
};

type LMTransaction = {
  id: number;
  date: string; // YYYY-MM-DD
  payee: string | null;
  amount: number; // positive number
  currency?: string | null;
  notes?: string | null;
  category_id: number | null;
  status?: string | null;
  is_group?: boolean;
  parent_id?: number | null;
  external_id?: string | null;
  plaid_metadata?:
    | string
    | {
        account_id?: string;
        account_owner?: string;
        transaction_id?: string;
        category?: string[];
        category_id?: string;
        merchant_name?: string | null;
        name?: string | null;
        counterparties?: Array<{
          confidence_level?: string;
          name?: string | null;
          type?: string | null;
        }>;
        iso_currency_code?: string | null;
        payment_channel?: string;
        payment_meta?: {
          by_order_of?: string | null;
          payee?: string | null;
          payer?: string | null;
          payment_method?: string | null;
          payment_processor?: string | null;
          ppd_id?: string | null;
          reason?: string | null;
          reference_number?: string | null;
        } | null;
        personal_finance_category?: {
          confidence_level?: string;
          primary?: string;
          detailed?: string;
        } | null;
        personal_finance_category_icon_url?: string | null;
        transaction_type?: string | null;
        pending?: boolean;
        location?: {
          address?: string;
          city?: string;
          region?: string;
          postal_code?: string;
          country?: string;
          lat?: number | null;
          lon?: number | null;
          store_number?: string | null;
        };
      }
    | null;
};

// Add a small type for suggestions
type CategorySuggestion = { name: string; justification?: string; confidence?: number | null };

// Provider/model presets
type ProviderId = "openai" | "anthropic" | "google";
type ModelConfig = { name: string; temperature: number };
const PROVIDERS: { id: ProviderId; name: string; models: ModelConfig[]; keyHint: string }[] = [
  {
    id: "openai",
    name: "OpenAI",
    models: [
      { name: "gpt-5-mini", temperature: 1 },
      { name: "gpt-4.1-mini", temperature: 0 },
      { name: "gpt-4o-mini", temperature: 0 },
      { name: "o4-mini", temperature: 0 },
      { name: "o3-mini", temperature: 0 },
    ],
    keyHint: "sk-...",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    models: [
      { name: "claude-3-5-haiku-latest", temperature: 0 },
      { name: "claude-4-sonnet-latest", temperature: 0 },
    ],
    keyHint: "anth-...",
  },
  {
    id: "google",
    name: "Google (Gemini)",
    models: [{ name: "gemini-2.5-flash", temperature: 0 }],
    keyHint: "AIza...",
  },
];

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

// Helper function to parse Plaid metadata
function parsePlaidMetadata(metadata: string | object | null | undefined): any {
  if (!metadata) return null;
  if (typeof metadata === "string") {
    try {
      return JSON.parse(metadata);
    } catch (e) {
      console.warn("Failed to parse Plaid metadata:", e);
      return null;
    }
  }
  return metadata as any;
}
const log = (msg: string, cls: "ok" | "warn" | "err" | "" = "") => {
  const el = $("#log");
  // Ensure assistive tech picks up new lines
  if (!el.getAttribute("role")) {
    el.setAttribute("role", "log");
    el.setAttribute("aria-live", "polite");
  }

  const line = document.createElement("div");
  line.className = `log-line${cls ? " " + cls : ""}`;

  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const level = document.createElement("span");
  level.className = `log-level ${cls || "info"}`;
  level.textContent =
    cls === "ok" ? "OK" : cls === "warn" ? "WARN" : cls === "err" ? "ERROR" : "INFO";

  const message = document.createElement("span");
  message.className = "log-msg";
  message.textContent = msg;

  line.appendChild(time);
  line.appendChild(level);
  line.appendChild(message);

  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
};

// Add: progress bar helper (prevents crash at run start)
function setBar(pct: number) {
  const el = document.querySelector("#bar") as HTMLElement | null;
  if (el) el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

// Add: small helpers for safe HTML and colored amount formatting
function esc(s: string | null | undefined): string {
  return (s ?? "").replace(
    /[&<>"']/g,
    ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]!
  );
}
function formatAmountHTML(amount: number, currency?: string | null): string {
  const code = currency || "USD";
  const fmt = new Intl.NumberFormat(undefined, { style: "currency", currency: code });
  const cls = amount < 0 ? "inc" : "exp"; // negative -> income/credit (green), else expense/debit (red)
  return `<span class="amt ${cls}">${fmt.format(amount)}</span>`;
}
// New: plain-text currency formatter for logs
function formatAmountText(amount: number, currency?: string | null): string {
  const code = currency || "USD";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: code }).format(amount);
}

// Apply theme to <html> via data-theme; "system" keeps CSS media query in control
type ThemeMode = "system" | "light" | "dark";
function applyTheme(theme: ThemeMode) {
  document.documentElement.setAttribute("data-theme", theme);
}
function nextTheme(t: ThemeMode): ThemeMode {
  return t === "system" ? "light" : t === "light" ? "dark" : "system";
}
function updateThemeToggleUI(btn: HTMLButtonElement, theme: ThemeMode) {
  const icon = theme === "dark" ? "🌙" : theme === "light" ? "☀️" : "🖥️";
  const title =
    theme === "dark" ? "Theme: Dark" : theme === "light" ? "Theme: Light" : "Theme: System";
  btn.textContent = icon;
  btn.title = title;
  btn.setAttribute("aria-label", title);
}

// --- Provider/model UI helpers ---
function populateProviderSelect() {
  const sel = $("#provider") as HTMLSelectElement;
  sel.innerHTML = "";
  for (const p of PROVIDERS) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
}
function populateModelSelect(providerId: ProviderId, preselect?: string) {
  const sel = $("#model") as HTMLSelectElement;
  sel.innerHTML = "";
  const p = PROVIDERS.find(x => x.id === providerId) ?? PROVIDERS[0];

  // Add predefined models
  for (const m of p.models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    if (preselect && preselect === m.name) opt.selected = true;
    sel.appendChild(opt);
  }

  // If there's a saved custom model, add it
  const savedCustomModel = localStorage.getItem("custom_model");
  if (savedCustomModel && savedCustomModel.trim()) {
    const customModelOpt = document.createElement("option");
    customModelOpt.value = savedCustomModel;
    customModelOpt.textContent = savedCustomModel;
    if (preselect === savedCustomModel) customModelOpt.selected = true;
    sel.appendChild(customModelOpt);
  }

  // Add custom option
  const customOpt = document.createElement("option");
  customOpt.value = "custom";
  customOpt.textContent = "Custom model...";
  sel.appendChild(customOpt);

  if (!preselect && sel.firstChild && sel instanceof HTMLSelectElement) {
    sel.selectedIndex = 0;
  }
}
function updateKeyPlaceholder(providerId: ProviderId) {
  const input = $("#openaiKey") as HTMLInputElement;
  const p = PROVIDERS.find(x => x.id === providerId) ?? PROVIDERS[0];
  input.placeholder = `${p.keyHint}`;
}

function getModelTemperature(providerId: ProviderId, modelName: string): number {
  const provider = PROVIDERS.find(x => x.id === providerId) ?? PROVIDERS[0];
  const model = provider.models.find(m => m.name === modelName);
  return model?.temperature ?? 0;
}

function setupModelSelectHandlers() {
  const modelSel = $("#model") as HTMLSelectElement;
  const customInput = $("#customModel") as HTMLInputElement;

  const handleModelChange = () => {
    if (modelSel.value === "custom") {
      customInput.style.display = "block";
      customInput.focus();
    } else {
      customInput.style.display = "none";
    }
  };

  const handleCustomModelInput = () => {
    const customModelName = customInput.value.trim();
    if (customModelName) {
      // Remove any existing custom model entries (except the "Custom model..." option)
      const provider = ($("#provider") as HTMLSelectElement).value as ProviderId;
      const existingCustomOptions = Array.from(modelSel.options).filter(
        opt =>
          opt.value !== "custom" &&
          !PROVIDERS.find(p => p.id === provider)?.models.find(m => m.name === opt.value)
      );
      existingCustomOptions.forEach(opt => opt.remove());

      // Add the new custom model as an option
      const customModelOpt = document.createElement("option");
      customModelOpt.value = customModelName;
      customModelOpt.textContent = customModelName;
      customModelOpt.selected = true;

      // Insert before the "Custom model..." option
      const customOption = Array.from(modelSel.options).find(opt => opt.value === "custom");
      if (customOption) {
        modelSel.insertBefore(customModelOpt, customOption);
      }

      // Hide the custom input and save to localStorage
      customInput.style.display = "none";
      localStorage.setItem("custom_model", customModelName);
      localStorage.setItem("llm_model", customModelName);
    }
  };

  customInput.addEventListener("blur", handleCustomModelInput);
  customInput.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      handleCustomModelInput();
    }
  });

  modelSel.addEventListener("change", handleModelChange);
  handleModelChange(); // Initialize state
}

function getSelectedModel(): string {
  const modelSel = $("#model") as HTMLSelectElement;
  const customInput = $("#customModel") as HTMLInputElement;

  if (modelSel.value === "custom") {
    return customInput.value.trim();
  }
  return modelSel.value;
}

// --- Lunch Money API helpers ---
function twoMonthsRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setMonth(start.getMonth() - 2);
  const fmt = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  return { start: fmt(start), end: fmt(end) };
}

async function lmFetch<T>(token: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${LM_BASE}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Lunch Money HTTP ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

async function getCategories(token: string): Promise<LMCategory[]> {
  const data = await lmFetch<{ categories: LMCategory[] }>(token, "/categories");
  return (data.categories || []).filter(c => !c.is_group && !c.archived);
}

async function getTransactions(
  token: string,
  start: string,
  end: string
): Promise<LMTransaction[]> {
  const params = new URLSearchParams({
    start_date: start,
    end_date: end,
    is_group: "false",
    limit: "500",
  });
  const data = await lmFetch<{ transactions: LMTransaction[] }>(
    token,
    `/transactions?${params.toString()}`
  );
  return (data.transactions || []).filter(t => !t.is_group && t.category_id === null);
}

async function updateTransactionCategory(
  token: string,
  id: number,
  categoryId: number
): Promise<void> {
  await lmFetch(token, `/transactions/${id}`, {
    method: "PUT",
    body: JSON.stringify({
      transaction: {
        category_id: categoryId,
      },
    }),
  });
}

// --- Prompt building ---
function buildSystemPrompt(categories: LMCategory[]): string {
  const categoriesWithIds = categories.map(c => {
    let categoryInfo = `"${c.name}" (ID: ${c.id})`;
    if (c.description && c.description.trim()) {
      categoryInfo += ` - ${c.description.trim()}`;
    }
    return categoryInfo;
  });

  return [
    "You categorize personal finance transactions.",
    "",
    "CRITICAL REQUIREMENTS:",
    "- You MUST ONLY choose category names from the EXACT list provided below",
    "- Category names must match EXACTLY - character for character, including punctuation and spacing",
    "- DO NOT create new categories, split compound categories, or use variations",
    "- DO NOT use similar-sounding names that aren't in the list",
    "- If unsure, pick the closest match from the provided list",
    "",
    "EXAMPLE: If the list contains 'Gas, Transportation', you must use 'Gas, Transportation' exactly - NEVER 'Gas' or 'Transportation' separately.",
    "",
    "Return ONLY valid JSON with this exact structure:",
    `{ "suggestions": [`,
    `  { "name": "<EXACT category name from the list below>", "justification": "<short reason>", "confidence": 0.85 }`,
    `] }`,
    "",
    "REQUIREMENTS:",
    "- Include 3 suggestions, sorted by confidence (highest first)",
    "- Category names must be EXACT matches from the list below",
    "- Confidence should be 0.0-1.0 (not percentages)",
    "- Keep justifications to one sentence",
    "",
    "AVAILABLE CATEGORIES (you must choose from this exact list):",
    "Note: Some categories include descriptions to help you understand their purpose.",
    "",
    categoriesWithIds.join("\n"),
    "",
    "Remember: Use the EXACT category name as shown above. Do not modify, abbreviate, or create variations.",
  ]
    .filter(Boolean)
    .join("\n");
}

function buildTransactionPrompt(t: LMTransaction): string {
  const plaidMeta = parsePlaidMetadata(t.plaid_metadata);

  const plaidCategory = plaidMeta?.category?.join(", ") || "Unknown";
  const loc = plaidMeta?.location;
  const location = [loc?.city, loc?.region].filter(Boolean).join(", ") || "Unknown";
  const merchant = plaidMeta?.merchant_name || plaidMeta?.name || t.payee || "Unknown";
  const pfcPrimary = plaidMeta?.personal_finance_category?.primary;
  const pfcDetailed = plaidMeta?.personal_finance_category?.detailed;
  const _pfc =
    pfcPrimary || pfcDetailed
      ? `${pfcPrimary ?? ""}${pfcPrimary && pfcDetailed ? " > " : ""}${pfcDetailed ?? ""}`
      : "Unknown";
  const _channel = plaidMeta?.payment_channel || "Unknown";
  const counterparties = plaidMeta?.counterparties || [];
  const allCounterparties = counterparties
    .map(
      (cp: { name: string; type: string; confidence_level: string }) =>
        `${cp.name} (${cp.type}, ${cp.confidence_level})`
    )
    .join("; ");
  const transactionType = plaidMeta?.transaction_type || "";
  const pfcConfidence = plaidMeta?.personal_finance_category?.confidence_level || "";
  const catId = plaidMeta?.category_id || "";
  const pending =
    plaidMeta?.pending === true ? "true" : plaidMeta?.pending === false ? "false" : "unknown";

  return [
    "Please categorize this transaction:",
    "",
    `- Payee: ${t.payee ?? "Unknown"}`,
    `- Merchant: ${merchant}`,
    `- Amount: ${t.amount}`,
    `- Currency: ${t.currency ?? plaidMeta?.iso_currency_code ?? ""}`,
    `- Date: ${t.date}`,
    `- Notes: ${t.notes ?? ""}`,
    `- Plaid Category: ${plaidCategory}${catId ? ` (#${catId})` : ""}`,
    `- Personal Finance Category: ${_pfc}${pfcConfidence ? ` (${pfcConfidence} confidence)` : ""}`,
    `- Payment Channel: ${_channel}`,
    transactionType ? `- Transaction Type: ${transactionType}` : undefined,
    allCounterparties
      ? `- ${counterparties.length === 1 ? "Counterparty" : "Counterparties"}: ${allCounterparties}`
      : undefined,
    `- Location: ${location}`,
    `- Pending: ${pending}`,
  ]
    .filter(Boolean)
    .join("\n");
}

// --- LangChain-powered suggestions ---
async function chooseCategoryOptions(
  provider: ProviderId,
  model: string,
  apiKey: string,
  systemPrompt: string,
  transactionPrompt: string
): Promise<CategorySuggestion[]> {
  let llm:
    | InstanceType<typeof ChatOpenAI>
    | InstanceType<typeof ChatAnthropic>
    | InstanceType<typeof ChatGoogleGenerativeAI>;

  const temperature = getModelTemperature(provider, model);

  if (provider === "openai") {
    llm = new ChatOpenAI({ apiKey, model, temperature });
  } else if (provider === "anthropic") {
    llm = new ChatAnthropic({ apiKey, model, temperature });
  } else {
    llm = new ChatGoogleGenerativeAI({ apiKey, model, temperature });
  }

  const result = await llm.invoke([
    {
      role: "system",
      content: systemPrompt,
    },
    { role: "user", content: transactionPrompt },
  ]);
  // log the JSON result
  const content = Array.isArray((result as { content?: unknown }).content)
    ? (result as { content: { text?: string }[] }).content
        .map((c: { text?: string }) => (typeof c?.text === "string" ? c.text : String(c ?? "")))
        .join("\n")
    : String((result as { content?: unknown }).content ?? "");

  const raw = content.trim();
  if (!raw) return [];

  // Reuse the existing robust JSON extraction
  const m = /```(?:json)?\s*([\s\S]*?)\s*```/i.exec(raw);
  const jsonText = (m ? m[1] : raw).trim();

  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch {
    const start = jsonText.indexOf("{");
    const end = jsonText.lastIndexOf("}");
    if (start >= 0 && end > start) {
      parsed = JSON.parse(jsonText.slice(start, end + 1));
    } else {
      return [];
    }
  }

  const arr: CategorySuggestion[] = Array.isArray((parsed as any)?.suggestions)
    ? (parsed as any).suggestions
    : [];
  return arr
    .map(s => ({
      name: String(s?.name ?? "").trim(),
      justification: s?.justification ? String(s.justification).trim() : undefined,
      confidence: typeof s?.confidence === "number" ? s.confidence : null,
    }))
    .filter(s => s.name.length > 0)
    .slice(0, 3);
}

// --- Category matching & modal helpers ---
function matchCategoryId(name: string, categories: LMCategory[]): number | null {
  // First try exact match (case-sensitive)
  let found = categories.find(c => c.name === name);
  if (found) return found.id;

  // Then try case-insensitive exact match
  found = categories.find(c => c.name.toLowerCase() === name.toLowerCase());
  if (found) return found.id;

  // Only use fuzzy matching as last resort, but warn about it
  const lo = name.toLowerCase();
  found = categories.find(
    c => c.name.toLowerCase().includes(lo) || lo.includes(c.name.toLowerCase())
  );
  if (found) {
    console.warn(
      `Category suggestion "${name}" did not match exactly. Using "${found.name}" instead.`
    );
  }
  return found ? found.id : null;
}

function validateCategorySuggestions(
  suggestions: CategorySuggestion[],
  categories: LMCategory[]
): CategorySuggestion[] {
  const validCategoryNames = new Set(categories.map(c => c.name));

  return suggestions.filter(suggestion => {
    const isValid = validCategoryNames.has(suggestion.name);
    if (!isValid) {
      console.warn(`Invalid category suggestion: "${suggestion.name}" is not in the category list`);
    }
    return isValid;
  });
}

function openModal() {
  $("#modal").classList.add("show");
  $("#modal").setAttribute("aria-hidden", "false");
}
function closeModal() {
  $("#modal").classList.remove("show");
  $("#modal").setAttribute("aria-hidden", "true");
  hideModalLoading(); // Clear loading state when modal closes
}
function showModalLoading(text = "Processing...") {
  const loadingEl = $("#modalLoading");
  const textEl = $("#loadingText");
  textEl.textContent = text;
  loadingEl.classList.add("show");
}
function hideModalLoading() {
  $("#modalLoading").classList.remove("show");
}
function clearSelect(select: HTMLSelectElement) {
  while (select.firstChild) select.removeChild(select.firstChild);
}
function populateCategorySelect(
  select: HTMLSelectElement,
  categories: LMCategory[],
  preselectId?: number | null
) {
  clearSelect(select);
  const optNone = document.createElement("option");
  optNone.value = "";
  optNone.textContent = "— Choose a category —";
  select.appendChild(optNone);

  const sorted = [...categories].sort((a, b) => a.name.localeCompare(b.name));
  for (const c of sorted) {
    const opt = document.createElement("option");
    opt.value = String(c.id);
    opt.textContent = c.name;
    if (preselectId && c.id === preselectId) opt.selected = true;
    select.appendChild(opt);
  }
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function logTransactionDetails(t: LMTransaction): string {
  const plaidMeta = parsePlaidMetadata(t.plaid_metadata);
  const merchant = plaidMeta?.merchant_name || plaidMeta?.name || t.payee || "Unknown";
  const pfcPrimary = plaidMeta?.personal_finance_category?.primary;
  const pfcDetailed = plaidMeta?.personal_finance_category?.detailed;
  const pfc =
    pfcPrimary || pfcDetailed
      ? `${pfcPrimary ?? ""}${pfcPrimary && pfcDetailed ? ">" : ""}${pfcDetailed ?? ""}`
      : "";
  const channel = plaidMeta?.payment_channel || "";
  const plaidCategory = plaidMeta?.category?.join(", ") || "";
  const currencyCode = t.currency ?? plaidMeta?.iso_currency_code ?? "USD";

  const base = `${esc(t.date)} • ${esc(merchant)} • ${formatAmountHTML(t.amount, currencyCode)}`;
  const extras = [
    plaidCategory ? `Plaid Category: ${esc(plaidCategory)}` : "",
    pfc ? `Personal Finance Category: ${esc(pfc)}` : "",
    channel ? `Channel: ${esc(channel)}` : "",
  ]
    .filter(Boolean)
    .join(" • ");
  return extras ? `${base} • ${extras}` : base;
}

function formatTransactionCard(t: LMTransaction): string {
  const plaidMeta = parsePlaidMetadata(t.plaid_metadata);
  const merchant = t.payee || plaidMeta?.merchant_name || plaidMeta?.name || "Unknown";
  const pfcPrimary = plaidMeta?.personal_finance_category?.primary;
  const pfcDetailed = plaidMeta?.personal_finance_category?.detailed;
  const _pfc =
    pfcPrimary || pfcDetailed
      ? `${pfcPrimary ?? ""}${pfcPrimary && pfcDetailed ? " > " : ""}${pfcDetailed ?? ""}`
      : "";
  const _channel = plaidMeta?.payment_channel || "";
  const plaidCategory = plaidMeta?.category?.join(", ") || "";
  const currencyCode = t.currency ?? plaidMeta?.iso_currency_code ?? "USD";
  const location = plaidMeta?.location;
  const _locationStr = location ? [location.city, location.region].filter(Boolean).join(", ") : "";

  let html = `
    <div class="tx-header">
      <div class="tx-merchant">${esc(merchant)}</div>
      <div class="tx-date">${esc(t.date)}</div>
    </div>
    <div class="tx-details">
      <div class="tx-detail-row">
        <span class="tx-detail-label">Amount:</span>
        <span class="tx-detail-value">${formatAmountHTML(t.amount, currencyCode)}</span>
      </div>`;

  if (plaidCategory) {
    html += `
      <div class="tx-detail-row">
        <span class="tx-detail-label">Plaid Category:</span>
        <span class="tx-detail-value">${esc(plaidCategory)}</span>
      </div>`;
  }

  if (t.notes && t.notes.trim()) {
    html += `
      <div class="tx-detail-row">
        <span class="tx-detail-label">Notes:</span>
        <span class="tx-detail-value" style="font-style: italic;">${esc(t.notes)}</span>
      </div>`;
  }

  html += `</div>`;

  return html;
}

async function promptUserForCategoryWithLoading(
  t: LMTransaction,
  categories: LMCategory[],
  provider: ProviderId,
  model: string,
  apiKey: string,
  systemPrompt: string
): Promise<number | null> {
  // Set up modal content first
  const txPreview = $("#txPreview");
  const suggestWrap = $("#suggestWrap");
  const select = $("#catSelect") as HTMLSelectElement;

  // Show transaction information immediately
  txPreview.innerHTML = formatTransactionCard(t);
  suggestWrap.textContent = "";
  populateCategorySelect(select, categories);

  // Open modal with transaction info visible
  openModal();

  // Show loading and get suggestions
  showModalLoading("");

  try {
    const transactionPrompt = buildTransactionPrompt(t);
    const rawSuggestions = await chooseCategoryOptions(
      provider,
      model,
      apiKey,
      systemPrompt,
      transactionPrompt
    );
    const suggestions = validateCategorySuggestions(rawSuggestions, categories);
    hideModalLoading();

    return await promptUserForCategory(t, categories, suggestions);
  } catch (error) {
    hideModalLoading();
    throw error;
  }
}

async function promptUserForCategory(
  t: LMTransaction,
  categories: LMCategory[],
  suggestions?: CategorySuggestion[]
): Promise<number | null> {
  const _modal = $("#modal");
  const suggestWrap = $("#suggestWrap");
  const select = $("#catSelect") as HTMLSelectElement;
  const btnSave = $("#saveBtn") as HTMLButtonElement;
  const btnSkip = $("#skipBtn") as HTMLButtonElement;
  const btnCancelModal = $("#cancelBtnModal") as HTMLButtonElement;

  // Clear suggestions and populate with new ones
  suggestWrap.textContent = "";
  const plaidMeta = parsePlaidMetadata(t.plaid_metadata);
  const iconUrl = plaidMeta?.personal_finance_category_icon_url || "";
  if (suggestions?.length) {
    for (const s of suggestions) {
      const id = matchCategoryId(s.name, categories);
      const row = document.createElement("div");
      row.className = "sugg";
      row.title = id ? "Click to select this category" : "Category not found in your list";
      row.setAttribute("role", "button");
      row.tabIndex = 0;

      if (iconUrl) {
        const img = document.createElement("img");
        img.className = "suggIcon";
        img.src = iconUrl;
        img.alt = "PFC icon";
        row.appendChild(img);
      }

      const body = document.createElement("div");
      body.className = "suggBody";
      const nameEl = document.createElement("span");
      nameEl.className = "name";
      nameEl.textContent = s.name;

      const rawConf =
        typeof s.confidence === "number" && Number.isFinite(s.confidence) ? s.confidence : null;
      const norm = rawConf === null ? null : rawConf > 1 ? rawConf / 100 : rawConf;
      if (norm !== null && norm >= 0) {
        const pct = Math.round(Math.max(0, Math.min(1, norm)) * 100);
        const level = pct >= 80 ? "high" : pct >= 50 ? "med" : "low";
        const confEl = document.createElement("span");
        confEl.className = `conf ${level}`;
        confEl.textContent = `${pct}%`;
        confEl.title = `Model confidence: ${pct}%`;
        nameEl.appendChild(confEl);
      }

      const justEl = document.createElement("span");
      justEl.className = "just";
      justEl.textContent = s.justification || "";
      body.appendChild(nameEl);
      body.appendChild(justEl);
      row.appendChild(body);

      const pick = () => {
        if (id) {
          select.value = String(id);
          row.style.outline = "2px solid #3b82f6";
          setTimeout(() => (row.style.outline = ""), 300);
        }
      };
      row.addEventListener("click", pick);
      row.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          pick();
        }
      });

      suggestWrap.appendChild(row);
    }
  }

  // Update select with the top suggestion preselected
  const preId = suggestions?.[0]?.name ? matchCategoryId(suggestions[0].name, categories) : null;
  if (preId) {
    select.value = String(preId);
  }

  // Don't open modal here - it's already open from promptUserForCategoryWithLoading
  // Avoid popping up keyboards on touch devices/small screens
  const isCoarse =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(pointer: coarse)").matches;
  if (!isCoarse && window.innerWidth > 640) {
    setTimeout(() => select.focus(), 0);
  }
  return new Promise<number | null>(resolve => {
    const onSave = async () => {
      const val = select.value.trim();
      if (val) {
        showModalLoading("Saving category...");
        resolve(Number(val));
      } else {
        resolve(null);
      }
      cleanup();
    };
    const onSkip = () => {
      resolve(null);
      cleanup();
    };
    const onCancel = () => {
      cancelled = true;
      resolve(null);
      cleanup();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
      if (e.key === "Enter") {
        e.preventDefault();
        onSave();
      }
    };
    function cleanup() {
      btnSave.removeEventListener("click", onSave);
      btnSkip.removeEventListener("click", onSkip);
      btnCancelModal.removeEventListener("click", onCancel);
      document.removeEventListener("keydown", onKey);
      // Don't close modal here - let the main loop handle it after saving
    }

    btnSave.addEventListener("click", onSave);
    btnSkip.addEventListener("click", onSkip);
    btnCancelModal.addEventListener("click", onCancel);
    document.addEventListener("keydown", onKey);

    setTimeout(() => select.focus(), 0);
  });
}

/* ---------- App bootstrapping ---------- */
function setDefaultsFromLocalStorage() {
  const lm = localStorage.getItem("lmToken") || "";
  const oa = localStorage.getItem("apiKey") || "";
  ($("#lmToken") as HTMLInputElement).value = lm;
  ($("#openaiKey") as HTMLInputElement).value = oa;

  // Provider/model defaults
  populateProviderSelect();
  const providerSel = $("#provider") as HTMLSelectElement;
  const _modelSel = $("#model") as HTMLSelectElement;
  const savedProv = (localStorage.getItem("llm_provider") as ProviderId) || "openai";
  const savedModel = localStorage.getItem("llm_model") || undefined;

  providerSel.value = savedProv;
  updateKeyPlaceholder(savedProv);
  populateModelSelect(savedProv, savedModel);

  providerSel.addEventListener("change", () => {
    const pid = providerSel.value as ProviderId;
    updateKeyPlaceholder(pid);
    populateModelSelect(pid);
    setupModelSelectHandlers();
  });

  setupModelSelectHandlers();

  // Theme: initialize toggle and bind cycle behavior
  const toggle = document.querySelector("#themeToggle") as HTMLButtonElement | null;
  const savedTheme = (localStorage.getItem("theme") as ThemeMode) || "system";
  applyTheme(savedTheme);
  if (toggle) {
    updateThemeToggleUI(toggle, savedTheme);
    toggle.addEventListener("click", () => {
      const curr = (document.documentElement.getAttribute("data-theme") as ThemeMode) || "system";
      const next = nextTheme(curr);
      applyTheme(next);
      localStorage.setItem("theme", next);
      updateThemeToggleUI(toggle, next);
    });
  }

  const { start, end } = twoMonthsRange();
  const startEl = $("#start") as HTMLInputElement;
  const endEl = $("#end") as HTMLInputElement;
  if (!startEl.value) startEl.value = start;
  if (!endEl.value) endEl.value = end;
}

async function run() {
  $("#log").textContent = "";
  setBar(0);

  const lmToken = ($("#lmToken") as HTMLInputElement).value.trim();
  const apiKey = ($("#openaiKey") as HTMLInputElement).value.trim();
  const provider = ($("#provider") as HTMLSelectElement).value as ProviderId;
  const model = getSelectedModel();
  const start = ($("#start") as HTMLInputElement).value;
  const end = ($("#end") as HTMLInputElement).value;

  // Reset cancel state and enable Cancel button
  cancelled = false;
  const runBtn = $("#runBtn") as HTMLButtonElement;
  const cancelBtn = $("#cancelBtn") as HTMLButtonElement;
  const onGlobalCancel = () => {
    if (!cancelled) {
      cancelled = true;
      log("Cancelling…", "warn");
      closeModal();
    }
  };
  cancelBtn.disabled = false;
  cancelBtn.addEventListener("click", onGlobalCancel, { once: true });

  if (!lmToken) {
    log("Missing Lunch Money token.", "err");
    cancelBtn.disabled = true;
    return;
  }
  if (!apiKey) {
    log("Missing model API key.", "err");
    cancelBtn.disabled = true;
    return;
  }
  if (!model) {
    log("Please select a model or enter a custom model name.", "err");
    cancelBtn.disabled = true;
    return;
  }

  localStorage.setItem("lmToken", lmToken);
  localStorage.setItem("apiKey", apiKey);
  localStorage.setItem("llm_provider", provider);
  localStorage.setItem("llm_model", model);

  // Also save custom model input value if used
  const customInput = $("#customModel") as HTMLInputElement;
  if (customInput.value.trim()) {
    localStorage.setItem("custom_model", customInput.value.trim());
  }

  try {
    log("Fetching categories...");
    const categories = await getCategories(lmToken);
    if (!categories.length) {
      log("No categories found.", "err");
      return;
    }
    log(`Found ${categories.length} active categories.`, "ok");

    log(`Fetching uncategorized transactions from ${start} to ${end}...`);
    const txs = await getTransactions(lmToken, start, end);
    if (!txs.length) {
      log("No uncategorized transactions found in the selected range.", "warn");
      return;
    }
    log(`Found ${txs.length} uncategorized transactions.`);

    // Build system prompt once for all transactions
    const systemPrompt = buildSystemPrompt(categories);
    let done = 0;
    for (const t of txs) {
      if (cancelled) {
        log("Cancelled by user.", "warn");
        break;
      }
      const plaidMeta = parsePlaidMetadata(t.plaid_metadata);
      const currencyCode = t.currency ?? plaidMeta?.iso_currency_code ?? "USD";
      const title = `${t.date} • ${t.payee ?? "Unknown"} • ${formatAmountText(t.amount, currencyCode)}`;
      try {
        log(`Suggesting category for: ${title}`);
        if (cancelled) {
          log("Cancelled by user.", "warn");
          break;
        }

        // Open modal first, then show loading while getting suggestions
        const chosenId = await promptUserForCategoryWithLoading(
          t,
          categories,
          provider,
          model,
          apiKey,
          systemPrompt
        );
        if (cancelled) {
          log("Cancelled by user.", "warn");
          closeModal(); // Ensure modal is closed when cancelled
          break;
        }
        if (!chosenId) {
          log(`Skipped: ${title}`, "warn");
          closeModal(); // Close modal when transaction is skipped
          done++;
          setBar((done / txs.length) * 100);
          continue;
        }

        try {
          await updateTransactionCategory(lmToken, t.id, chosenId);
          const catName = categories.find(c => c.id === chosenId)?.name ?? String(chosenId);
          log(`Updated ${title} -> ${catName}`, "ok");
        } finally {
          hideModalLoading();
          closeModal();
        }
      } catch (e: unknown) {
        log(`Error on ${title}: ${e instanceof Error ? e.message : String(e)}`, "err");
      } finally {
        done++;
        setBar((done / txs.length) * 100);
      }
    }

    if (!cancelled) log("Finished.");
  } catch (e: unknown) {
    log(`Fatal error: ${e instanceof Error ? e.message : String(e)}`, "err");
  } finally {
    // Restore buttons
    cancelBtn.disabled = true;
    // Ensure we can cancel again in the next run
    cancelBtn.replaceWith(cancelBtn.cloneNode(true) as HTMLButtonElement);
    // Rebind run button disabled state like before
    runBtn.disabled = false;
  }
}

function main() {
  setDefaultsFromLocalStorage();
  $("#runBtn").addEventListener("click", async e => {
    const btn = e.currentTarget as HTMLButtonElement;
    btn.disabled = true;
    try {
      await run();
    } finally {
      // run() handles reenabling; keep here as safety
      btn.disabled = false;
    }
  });
}

main();
