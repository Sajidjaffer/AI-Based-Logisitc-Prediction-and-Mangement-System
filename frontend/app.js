/**
 * AURA | AI Supply Chain Risk Intelligence
 * Clean, Simple, Story-Driven Application Controller
 */

// Application State
const state = {
  overview: null,
  mlModel: null,
  transfers: [],
  currentSku: null,
  allSkusList: [],
  catalogFilters: {
    q: '',
    risk_tier: 'ALL',
    category: 'ALL',
    page: 1,
    page_size: 20
  },
  catalogPagination: {
    total: 0,
    page: 1,
    total_pages: 1
  },
  charts: {
    forecastChart: null,
    burndownChart: null
  }
};

const formatCurrency = (val) => {
  if (val === undefined || val === null) return '$0';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(val);
};

const formatNumber = (val) => {
  if (val === undefined || val === null) return '0';
  return new Intl.NumberFormat('en-US').format(val);
};

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  setupThemeToggle();
  setupBriefingModal();
  setupPredictionExplorerEvents();
  setupWhatIfEvents();
  setupCatalogTableEvents();

  await loadInitialData();
});

async function loadInitialData() {
  try {
    const [overviewRes, mlRes, transfersRes, sampleSkusRes] = await Promise.all([
      fetch('/api/overview').then(r => r.json()),
      fetch('/api/ml-model').then(r => r.json()),
      fetch('/api/transfers').then(r => r.json()),
      fetch('/api/skus?page=1&page_size=50').then(r => r.json())
    ]);

    state.overview = overviewRes;
    state.mlModel = mlRes;
    state.transfers = transfersRes;
    state.allSkusList = sampleSkusRes.items || [];

    renderTopKPIs();
    populateProductDropdown(state.allSkusList);
    renderTopPriorityActions();
    loadCatalogTable();

    // Load first critical SKU into the Prediction Explorer by default
    const firstCritical = state.allSkusList.find(s => s.risk_tier === 'CRITICAL') || state.allSkusList[0];
    if (firstCritical) {
      loadSkuIntoExplorer(firstCritical.sku_id);
    }
  } catch (err) {
    console.error('Failed to load initial data:', err);
    showToast('Failed to connect to backend server. Make sure Python server is running.', 'error');
  }
}

// ============================================================================
// Top 4 KPI Metrics
// ============================================================================
function renderTopKPIs() {
  const o = state.overview;
  const ml = state.mlModel;
  if (!o) return;

  document.getElementById('kpi-total-loss').textContent = formatCurrency(o.total_lost_sales_30d);
  document.getElementById('kpi-loss-7d').textContent = formatCurrency(o.total_lost_sales_7d);
  document.getElementById('kpi-stockouts-count').textContent = formatNumber(o.critical_stockouts_7d);
  document.getElementById('kpi-savings-secured').textContent = formatCurrency(o.total_savings_secured || 0);
  document.getElementById('kpi-actions-executed').textContent = formatNumber(o.executed_actions_count || 0);

  if (ml && ml.metrics) {
    document.getElementById('kpi-model-r2').textContent = `R² ${ml.metrics.r2_score}`;
  }
}

// ============================================================================
// SECTION 1: The Interactive AI Prediction Explorer
// ============================================================================
function populateProductDropdown(skus) {
  const select = document.getElementById('select-sku');
  if (!select) return;

  // Curate unique products
  const uniqueItems = [];
  const seen = new Set();
  skus.forEach(s => {
    if (!seen.has(s.item_id)) {
      seen.add(s.item_id);
      uniqueItems.push(s);
    }
  });

  select.innerHTML = uniqueItems.slice(0, 35).map(s => `
    <option value="${s.item_id}">${s.product_name} (${s.cat_id})</option>
  `).join('');
}

function setupPredictionExplorerEvents() {
  const skuSelect = document.getElementById('select-sku');
  const facilitySelect = document.getElementById('select-facility');

  const onSelectionChange = () => {
    const itemId = skuSelect.value;
    const storeId = facilitySelect.value;
    const skuId = `${storeId}_${itemId}`;
    loadSkuIntoExplorer(skuId);
  };

  skuSelect.addEventListener('change', onSelectionChange);
  facilitySelect.addEventListener('change', onSelectionChange);

  // Quick Demo Preset Buttons
  document.getElementById('demo-btn-critical').addEventListener('click', (e) => {
    setPresetActive(e.target);
    const criticalSku = state.allSkusList.find(s => s.stockout_day && s.stockout_day <= 5);
    if (criticalSku) {
      facilitySelect.value = criticalSku.store_id;
      skuSelect.value = criticalSku.item_id;
      loadSkuIntoExplorer(criticalSku.sku_id);
    }
  });

  document.getElementById('demo-btn-spike').addEventListener('click', (e) => {
    setPresetActive(e.target);
    const spikeSku = state.allSkusList.find(s => s.has_demand_spike);
    if (spikeSku) {
      facilitySelect.value = spikeSku.store_id;
      skuSelect.value = spikeSku.item_id;
      loadSkuIntoExplorer(spikeSku.sku_id);
    }
  });

  document.getElementById('demo-btn-healthy').addEventListener('click', (e) => {
    setPresetActive(e.target);
    const healthySku = state.allSkusList.find(s => s.risk_tier === 'HEALTHY' || s.runway_days > 24);
    if (healthySku) {
      facilitySelect.value = healthySku.store_id;
      skuSelect.value = healthySku.item_id;
      loadSkuIntoExplorer(healthySku.sku_id);
    }
  });

  // Approve selected action button
  const approveBtn = document.getElementById('btn-approve-selected-action');
  approveBtn.addEventListener('click', async () => {
    if (!state.currentSku || !state.currentSku.recommended_action) return;
    const act = state.currentSku.recommended_action;
    await executeAction(act, approveBtn);
  });
}

function setPresetActive(activeBtn) {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  activeBtn.classList.add('active');
}

async function loadSkuIntoExplorer(skuId) {
  try {
    const sku = await fetch(`/api/sku/${skuId}`).then(r => r.json());
    state.currentSku = sku;

    // Update facility & product selectors if changed externally
    const skuSelect = document.getElementById('select-sku');
    const facilitySelect = document.getElementById('select-facility');
    if (skuSelect && skuSelect.value !== sku.item_id) {
      let optExists = Array.from(skuSelect.options).some(o => o.value === sku.item_id);
      if (!optExists) {
        skuSelect.add(new Option(`${sku.product_name} (${sku.cat_id})`, sku.item_id));
      }
      skuSelect.value = sku.item_id;
    }
    if (facilitySelect) facilitySelect.value = sku.store_id;

    // Render Step 1: Forecast Chart & Summary
    renderExplorerForecast(sku);

    // Render Step 2: Burndown Chart & Stockout Summary
    renderExplorerBurndown(sku);

    // Render Step 3: Recommended Action Box
    renderExplorerAction(sku);

  } catch (err) {
    console.error(`Failed to load SKU ${skuId}:`, err);
  }
}

function renderExplorerForecast(sku) {
  const ctx = document.getElementById('productForecastChart');
  if (!ctx) return;

  const hist = sku.historical_sales_30d || [];
  const fore = sku.forecast_demand_28d || [];

  const labels = [];
  const histData = [];
  const foreData = [];

  for (let i = 15; i >= 1; i--) {
    labels.push(`-${i}d`);
    histData.push(hist[30 - i] || 0);
    foreData.push(null);
  }
  for (let i = 1; i <= 28; i++) {
    labels.push(`+${i}d`);
    histData.push(null);
    foreData.push(fore[i - 1] || 0);
  }
  // Connect boundary
  foreData[14] = histData[14];

  if (state.charts.forecastChart) state.charts.forecastChart.destroy();

  state.charts.forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Past Sales',
          data: histData,
          borderColor: '#94a3b8',
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: 'AI Forecast',
          data: foreData,
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.08)',
          fill: true,
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.25
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.dataset.label}: ${c.parsed.y} units`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 }, maxTicksLimit: 8 },
          grid: { display: false }
        },
        y: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(0,0,0,0.05)' }
        }
      }
    }
  });

  const subtextEl = document.getElementById('story-forecast-subtext');
  const statEl = document.getElementById('story-forecast-stat');

  if (sku.has_demand_spike && sku.spike_events?.length) {
    const sp = sku.spike_events[0];
    subtextEl.innerHTML = `Model predicts an average demand of <strong>${sku.avg_daily_demand} units/day</strong>, with a holiday surge of <strong>+${sp.spike_pct}%</strong> on Day ${sp.day_index} (${sp.reason}).`;
    statEl.innerHTML = `<strong>Predicted Demand Surge:</strong> +${sp.spike_pct}% increase on Day ${sp.day_index}`;
  } else {
    subtextEl.innerHTML = `Model predicts steady baseline demand of <strong>${sku.avg_daily_demand} units/day</strong> with standard weekend retail velocity.`;
    statEl.innerHTML = `<strong>Steady Velocity:</strong> ${sku.avg_daily_demand} units projected per day`;
  }
}

function renderExplorerBurndown(sku) {
  const ctx = document.getElementById('productBurndownChart');
  if (!ctx) return;

  const burndown = sku.projected_inventory_28d || [];
  const labels = burndown.map((_, i) => `+${i + 1}d`);

  if (state.charts.burndownChart) state.charts.burndownChart.destroy();

  const isStockout = sku.stockout_day !== null && sku.stockout_day <= 28;

  state.charts.burndownChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'On-Hand Stock',
          data: burndown,
          borderColor: isStockout ? '#e11d48' : '#059669',
          backgroundColor: isStockout ? 'rgba(225, 29, 72, 0.08)' : 'rgba(5, 150, 105, 0.08)',
          fill: true,
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 }, maxTicksLimit: 8 },
          grid: { display: false }
        },
        y: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(0,0,0,0.05)' }
        }
      }
    }
  });

  const subtextEl = document.getElementById('story-burndown-subtext');
  const statEl = document.getElementById('story-burndown-stat');

  if (isStockout) {
    subtextEl.innerHTML = `Current on-hand inventory (<strong>${sku.on_hand_inventory} units</strong>) will completely run out on <strong>Day ${sku.stockout_day}</strong>.`;
    statEl.className = 'step-stat-callout danger';
    statEl.innerHTML = `<strong>Critical Stockout in ${sku.runway_days} Days:</strong> Estimated lost revenue is <strong>${formatCurrency(sku.potential_lost_sales_30d)}</strong>.`;
  } else {
    subtextEl.innerHTML = `On-hand inventory (<strong>${sku.on_hand_inventory} units</strong>) maintains a healthy <strong>${sku.runway_days} days</strong> runway buffer.`;
    statEl.className = 'step-stat-callout';
    statEl.innerHTML = `<strong>Buffer Healthy:</strong> No stockout predicted within 28 days.`;
  }
}

function renderExplorerAction(sku) {
  const actionBox = document.getElementById('story-action-box');
  const approveBtn = document.getElementById('btn-approve-selected-action');
  if (!actionBox || !approveBtn) return;

  const act = sku.recommended_action;
  if (!act || act.action_type === 'STANDARD_OPS') {
    actionBox.innerHTML = `
      <div class="action-header-badge">OPTIMAL INVENTORY</div>
      <div class="action-title-text">No Emergency Intervention Required</div>
      <div class="action-details-text">
        This product maintains sufficient safety stock buffer. Normal automated replenishment will trigger on schedule.
      </div>
    `;
    approveBtn.disabled = true;
    approveBtn.innerHTML = '&#10003; Inventory Healthy';
    approveBtn.className = 'btn btn-outline btn-block btn-lg';
    return;
  }

  const isExecuted = act.status === 'EXECUTED';
  const isTransfer = act.action_type === 'INTER_WAREHOUSE_TRANSFER';
  const isExpedite = act.action_type === 'EXPEDITED_SUPPLIER_PO';

  let badgeText = isTransfer ? 'INTER-WAREHOUSE TRANSFER' : isExpedite ? 'EXPEDITED PURCHASE ORDER' : 'SUPPLIER ESCALATION';
  let detailsText = '';

  if (isTransfer) {
    detailsText = `Transfer <strong>${act.transfer_qty} units</strong> from sister hub <strong>${act.source_store_id}</strong> (${act.source_donor_runway_before}d buffer) to <strong>${sku.store_id}</strong> with 24h transit time.`;
  } else if (isExpedite) {
    detailsText = `Issue emergency order of <strong>${act.order_qty} units</strong> to <strong>${act.supplier_name}</strong> via dedicated freight, cutting lead time to ${act.expedited_lead_time_days} days.`;
  } else {
    detailsText = `Dispatch priority SLA breach warning for in-transit order <strong>${act.po_number}</strong> (${act.delay_days} days delayed).`;
  }

  actionBox.innerHTML = `
    <div class="action-header-badge">${badgeText}</div>
    <div class="action-title-text">${act.title}</div>
    <div class="action-details-text">${detailsText}</div>
    <div class="action-financial-metrics">
      <div class="fin-metric-item">
        <span class="fin-label">Revenue Saved</span>
        <span class="fin-val">+${formatCurrency(act.protected_revenue || act.impact_at_risk || 0)}</span>
      </div>
      <div class="fin-metric-item">
        <span class="fin-label">Transport Cost</span>
        <span class="fin-val" style="color: var(--text-muted);">${formatCurrency(act.freight_cost || 0)}</span>
      </div>
      <div class="fin-metric-item">
        <span class="fin-label">Net Gain</span>
        <span class="fin-val text-success">+${formatCurrency(act.net_financial_gain || act.impact_at_risk || 0)}</span>
      </div>
    </div>
  `;

  if (isExecuted) {
    approveBtn.disabled = true;
    approveBtn.innerHTML = '&#10003; Mitigation Executed &amp; Dispatched';
    approveBtn.className = 'btn btn-outline btn-block btn-lg';
  } else {
    approveBtn.disabled = false;
    approveBtn.innerHTML = `&#10003; Approve &amp; Execute This Fix (+${formatCurrency(act.net_financial_gain || act.impact_at_risk || 0)})`;
    approveBtn.className = 'btn btn-success btn-block btn-lg';
  }
}

// ============================================================================
// SECTION 2: Top Priority Actions
// ============================================================================
function renderTopPriorityActions() {
  const container = document.getElementById('top-actions-container');
  if (!container) return;

  const topItems = (state.transfers || []).slice(0, 5);
  if (!topItems.length) {
    container.innerHTML = `<div class="text-center text-muted py-4">No pending urgent actions.</div>`;
    return;
  }

  container.innerHTML = topItems.map(act => {
    const isExecuted = act.status === 'EXECUTED';
    return `
      <div class="action-row-card ${isExecuted ? 'executed' : ''}" id="action-row-${act.id}">
        <div>
          <div class="product-cell-name">${act.product_name}</div>
          <div class="product-cell-facility">Destination: <strong>${act.target_store_id}</strong> &bull; Source: <strong>${act.source_store_id}</strong></div>
        </div>

        <div>
          <span class="metric-cell-label">Transfer Volume</span>
          <span class="metric-cell-val">${formatNumber(act.transfer_qty)} Units</span>
        </div>

        <div>
          <span class="metric-cell-label">Transit Time</span>
          <span class="metric-cell-val">${act.transit_days} Day Delivery</span>
        </div>

        <div>
          <span class="metric-cell-label">Protected Revenue</span>
          <span class="metric-cell-val success">+${formatCurrency(act.protected_revenue)}</span>
        </div>

        <div>
          <button class="btn btn-sm ${isExecuted ? 'btn-outline' : 'btn-success'} btn-block btn-row-approve"
                  data-action-id="${act.id}"
                  data-action-type="${act.action_type}"
                  data-sku-id="${act.sku_id}"
                  data-title="${act.title}"
                  data-savings="${act.net_financial_gain}"
                  data-qty="${act.transfer_qty}"
                  ${isExecuted ? 'disabled' : ''}>
            ${isExecuted ? '&#10003; Dispatched' : '&#10003; Approve Transfer'}
          </button>
        </div>
      </div>
    `;
  }).join('');

  // Attach button click listeners
  container.querySelectorAll('.btn-row-approve').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const actionObj = {
        id: btn.getAttribute('data-action-id'),
        action_type: btn.getAttribute('data-action-type'),
        sku_id: btn.getAttribute('data-sku-id'),
        title: btn.getAttribute('data-title'),
        net_financial_gain: parseFloat(btn.getAttribute('data-savings') || 0),
        transfer_qty: parseInt(btn.getAttribute('data-qty') || 0)
      };
      await executeAction(actionObj, btn);
    });
  });
}

async function executeAction(actionObj, buttonElement) {
  try {
    buttonElement.disabled = true;
    buttonElement.textContent = 'Authorizing...';

    const res = await fetch('/api/action/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action_id: actionObj.id,
        action_type: actionObj.action_type,
        sku_id: actionObj.sku_id,
        title: actionObj.title,
        net_savings: actionObj.net_financial_gain,
        transfer_qty: actionObj.transfer_qty
      })
    }).then(r => r.json());

    if (res.success) {
      buttonElement.innerHTML = '&#10003; Dispatched';
      buttonElement.className = 'btn btn-outline btn-sm';

      // Update KPIs
      document.getElementById('kpi-savings-secured').textContent = formatCurrency(res.total_savings_secured);
      document.getElementById('kpi-actions-executed').textContent = formatNumber(res.total_actions_executed);

      showToast(`Authorized! Secured ${formatCurrency(actionObj.net_financial_gain)} in protected revenue.`, 'success');

      // Update state if matches current SKU in explorer
      if (state.currentSku && state.currentSku.sku_id === actionObj.sku_id) {
        state.currentSku.recommended_action.status = 'EXECUTED';
        renderExplorerAction(state.currentSku);
      }
    }
  } catch (err) {
    console.error('Failed to execute action:', err);
    buttonElement.disabled = false;
    buttonElement.textContent = 'Retry Authorization';
    showToast('Failed to authorize action. Please retry.', 'error');
  }
}

// ============================================================================
// SECTION 3: Simple What-If Simulation
// ============================================================================
function setupWhatIfEvents() {
  const surgeSlider = document.getElementById('sim-surge-slider');
  const delaySlider = document.getElementById('sim-delay-slider');
  const surgeLabel = document.getElementById('sim-surge-label');
  const delayLabel = document.getElementById('sim-delay-label');
  const runBtn = document.getElementById('btn-run-simple-sim');

  surgeSlider.addEventListener('input', (e) => {
    surgeLabel.textContent = `+${e.target.value}%`;
  });

  delaySlider.addEventListener('input', (e) => {
    delayLabel.textContent = `+${e.target.value} Days`;
  });

  runBtn.addEventListener('click', async () => {
    const surge = parseFloat(surgeSlider.value);
    const delay = parseInt(delaySlider.value);

    try {
      runBtn.textContent = 'Calculating...';
      runBtn.disabled = true;

      const res = await fetch('/api/simulate-what-if', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          supplier_delay_days: delay,
          demand_surge_pct: surge,
          category: 'ALL',
          warehouse: 'ALL'
        })
      }).then(r => r.json());

      document.getElementById('sim-old-stockouts').textContent = formatNumber(res.baseline.critical_stockouts_7d);
      document.getElementById('sim-new-stockouts').textContent = formatNumber(res.simulated.critical_stockouts_7d);
      document.getElementById('sim-delta-stockouts').textContent = `${res.delta.critical_stockouts_change >= 0 ? '+' : ''}${formatNumber(res.delta.critical_stockouts_change)} SKUs`;

      document.getElementById('sim-old-loss').textContent = formatCurrency(res.baseline.total_lost_sales_30d);
      document.getElementById('sim-new-loss').textContent = formatCurrency(res.simulated.lost_sales_30d);
      document.getElementById('sim-delta-loss').textContent = `${res.delta.lost_sales_change >= 0 ? '+' : ''}${formatCurrency(res.delta.lost_sales_change)}`;

      document.getElementById('sim-advisory-message').textContent = res.executive_recommendation;

      showToast('Stress test recalculation complete.', 'success');
    } catch (err) {
      console.error('Failed to run simulation:', err);
    } finally {
      runBtn.textContent = 'Recalculate Network Stress Test';
      runBtn.disabled = false;
    }
  });
}

// ============================================================================
// SECTION 4: Product Catalog Table
// ============================================================================
function setupCatalogTableEvents() {
  const searchInput = document.getElementById('table-search-input');
  const riskFilter = document.getElementById('table-filter-risk');
  const catFilter = document.getElementById('table-filter-category');
  let debounce = null;

  searchInput.addEventListener('input', (e) => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      state.catalogFilters.q = e.target.value;
      state.catalogFilters.page = 1;
      loadCatalogTable();
    }, 250);
  });

  riskFilter.addEventListener('change', (e) => {
    state.catalogFilters.risk_tier = e.target.value;
    state.catalogFilters.page = 1;
    loadCatalogTable();
  });

  catFilter.addEventListener('change', (e) => {
    state.catalogFilters.category = e.target.value;
    state.catalogFilters.page = 1;
    loadCatalogTable();
  });

  document.getElementById('clean-btn-prev').addEventListener('click', () => {
    if (state.catalogFilters.page > 1) {
      state.catalogFilters.page--;
      loadCatalogTable();
    }
  });

  document.getElementById('clean-btn-next').addEventListener('click', () => {
    if (state.catalogFilters.page < state.catalogPagination.total_pages) {
      state.catalogFilters.page++;
      loadCatalogTable();
    }
  });
}

async function loadCatalogTable() {
  const params = new URLSearchParams({
    page: state.catalogFilters.page,
    page_size: state.catalogFilters.page_size,
    sort_by: 'potential_lost_sales_30d',
    sort_order: 'desc'
  });

  if (state.catalogFilters.q) params.append('q', state.catalogFilters.q);
  if (state.catalogFilters.risk_tier !== 'ALL') params.append('risk_tier', state.catalogFilters.risk_tier);
  if (state.catalogFilters.category !== 'ALL') params.append('category', state.catalogFilters.category);

  try {
    const res = await fetch(`/api/skus?${params.toString()}`).then(r => r.json());
    state.catalogPagination.total = res.total;
    state.catalogPagination.page = res.page;
    state.catalogPagination.total_pages = res.total_pages;

    document.getElementById('catalog-count-indicator').textContent = `Showing ${formatNumber(res.total)} Products`;
    document.getElementById('clean-page-label').textContent = `Page ${res.page} of ${res.total_pages || 1}`;

    const tbody = document.getElementById('clean-table-body');
    if (!res.items.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted py-4">No matching products found.</td></tr>`;
      return;
    }

    tbody.innerHTML = res.items.map(sku => {
      let badgeClass = 'neutral';
      if (sku.risk_tier === 'CRITICAL') badgeClass = 'danger';
      else if (sku.risk_tier === 'HIGH') badgeClass = 'warning';
      else if (sku.risk_tier === 'HEALTHY') badgeClass = 'success';

      const actionTitle = sku.recommended_action ? sku.recommended_action.title : 'Normal Replenishment';

      return `
        <tr onclick="onCatalogRowClick('${sku.sku_id}')">
          <td>
            <strong>${sku.product_name}</strong>
            <div style="font-size: 11px; color: var(--text-muted);">${sku.sku_id} &bull; ${sku.cat_id}</div>
          </td>
          <td><span class="badge-pill neutral">${sku.store_id}</span></td>
          <td style="font-family: var(--font-mono);">${formatCurrency(sku.unit_price)}</td>
          <td style="font-family: var(--font-mono);">${formatNumber(sku.on_hand_inventory)}</td>
          <td style="font-family: var(--font-mono);">${sku.avg_daily_demand} u/d</td>
          <td style="font-family: var(--font-mono); font-weight: 700; ${sku.runway_days < 7 ? 'color: var(--color-danger);' : ''}">${sku.runway_days}d</td>
          <td><span class="badge-pill ${badgeClass}">${sku.stockout_day ? `Day ${sku.stockout_day}` : '&gt; 28 Days'}</span></td>
          <td style="font-family: var(--font-mono); font-weight: 700; color: var(--color-danger);">${formatCurrency(sku.potential_lost_sales_30d)}</td>
          <td><span style="font-size: 12px; font-weight: 600; color: #0284c7;">${actionTitle}</span></td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load catalog table:', err);
  }
}

function onCatalogRowClick(skuId) {
  loadSkuIntoExplorer(skuId);
  window.scrollTo({ top: 380, behavior: 'smooth' });
}

// ============================================================================
// Modal, Toast & Theme Utilities
// ============================================================================
function setupBriefingModal() {
  const modal = document.getElementById('briefing-modal');
  const openBtn = document.getElementById('btn-open-briefing');
  const closeBtn = document.getElementById('btn-close-briefing');
  const copyBtn = document.getElementById('btn-copy-briefing');

  openBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/ai-report').then(r => r.json());
      document.getElementById('briefing-markdown-body').textContent = res.report_markdown;
      modal.classList.remove('d-none');
    } catch (err) {
      console.error(err);
    }
  });

  closeBtn.addEventListener('click', () => modal.classList.add('d-none'));

  copyBtn.addEventListener('click', () => {
    const text = document.getElementById('briefing-markdown-body').textContent;
    navigator.clipboard.writeText(text);
    showToast('Briefing copied to clipboard.', 'success');
  });
}

function setupThemeToggle() {
  const btn = document.getElementById('btn-theme-toggle');
  const text = document.getElementById('theme-btn-text');

  btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    text.textContent = next === 'dark' ? 'Light Mode' : 'Dark Mode';
  });
}

function showToast(msg, type = 'info') {
  const tray = document.getElementById('toast-tray');
  if (!tray) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  tray.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
