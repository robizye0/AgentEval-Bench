const fs = require('node:fs');

const DEFAULT_MARKER = '<!-- agenteval-bot -->';

function escapeCell(value) {
  return String(value ?? '').replaceAll('|', '\\|').replaceAll('\n', ' ');
}

function metric(caseResult, key) {
  const details = caseResult.details || {};
  if (key === 'cost') return Number(caseResult.total_cost_usd ?? details.cost_usd ?? 0);
  if (key === 'tokens') return Number(details.token_count ?? caseResult.token_count ?? 0);
  return Number(details.latency_ms ?? caseResult.latency_ms ?? 0);
}

function accuracy(caseResult) {
  return Number(caseResult.deterministic_accuracy_score ?? caseResult.tool_accuracy_score ?? 0);
}

function averageAccuracy(report) {
  if (Number.isFinite(report.average_deterministic_accuracy)) {
    return Number(report.average_deterministic_accuracy);
  }
  const cases = report.cases || [];
  return cases.length ? cases.reduce((sum, item) => sum + accuracy(item), 0) / cases.length : 0;
}

function regression(label, current, previous, threshold) {
  if (current <= previous) return null;
  if (previous <= 0) return `${label} +∞% (new usage)`;
  const delta = ((current - previous) / previous) * 100;
  return delta >= threshold ? `${label} +${delta.toFixed(0)}%` : null;
}

function buildComment(current, baseline = null, thresholds = {}, marker = DEFAULT_MARKER) {
  const limits = { tokens: 20, cost: 20, latency: 20, ...thresholds };
  const baseCases = new Map((baseline?.cases || []).map((item) => [String(item.id), item]));
  const caseRows = (current.cases || []).map((item) => {
    const id = String(item.id ?? item.details?.case_id ?? 'case');
    const base = baseCases.get(id);
    const delta = base ? `${((accuracy(item) - accuracy(base)) * 100 >= 0 ? '+' : '')}${((accuracy(item) - accuracy(base)) * 100).toFixed(1)} pp` : '—';
    const alerts = [];
    if (base) {
      for (const key of ['tokens', 'cost', 'latency']) {
        const alert = regression(key, metric(item, key), metric(base, key), Number(limits[key]));
        if (alert) alerts.push(alert);
      }
    }
    const status = item.passed ? '✅ PASS' : '❌ FAIL';
    const tokens = metric(item, 'tokens').toLocaleString('en-US');
    const cost = `$${metric(item, 'cost').toFixed(4)}`;
    const latency = `${metric(item, 'latency').toFixed(0)} ms`;
    return `| ${escapeCell(id)} | ${status} | ${(accuracy(item) * 100).toFixed(1)}% | ${delta} | ${tokens} | ${cost} | ${latency} | ${escapeCell(alerts.join(', ') || '—')} |`;
  });

  const passed = current.passed === true;
  const currentAverage = averageAccuracy(current);
  const baseAverage = baseline ? averageAccuracy(baseline) : null;
  const overallDelta = baseAverage === null
    ? 'baseline unavailable'
    : `${((currentAverage - baseAverage) * 100 >= 0 ? '+' : '')}${((currentAverage - baseAverage) * 100).toFixed(1)} pp vs main`;
  const table = [
    '| Case | Status | Accuracy | Δ vs main | Tokens | Cost | Latency | Regression alerts |',
    '|---|---:|---:|---:|---:|---:|---:|---|',
    ...caseRows,
  ].join('\n');
  const alertsExist = caseRows.some((row) => !row.endsWith('| — |'));
  return [
    marker,
    `## AgentEval ${passed ? '✅ PASS' : '❌ FAIL'}`,
    `**Mean deterministic accuracy:** ${(currentAverage * 100).toFixed(1)}% (${overallDelta})`,
    baseline ? `**Regression thresholds:** tokens ${limits.tokens}%, cost ${limits.cost}%, latency ${limits.latency}%` : '**No baseline report supplied; regression deltas are unavailable.**',
    '',
    table,
    '',
    alertsExist ? '> ⚠️ One or more cases exceeded a configured token, cost, or latency regression threshold.' : '> No token, cost, or latency regressions detected.',
    '',
    '_Updated automatically by AgentEval-Bench._',
  ].join('\n');
}

async function postOrUpdateComment({ github, context, current, baseline = null, thresholds = {}, marker = DEFAULT_MARKER }) {
  const issueNumber = context.issue?.number;
  if (!issueNumber || !context.repo?.owner || !context.repo?.repo) return { skipped: true };
  const body = buildComment(current, baseline, thresholds, marker);
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: issueNumber,
    per_page: 100,
  });
  const existing = comments.find((item) => typeof item.body === 'string' && item.body.includes(marker));
  if (existing) {
    await github.rest.issues.updateComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      comment_id: existing.id,
      body,
    });
    return { updated: true, commentId: existing.id };
  }
  const response = await github.rest.issues.createComment({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: issueNumber,
    body,
  });
  return { created: true, commentId: response.data.id };
}

function readReport(filePath, label) {
  if (!filePath) return null;
  if (!fs.existsSync(filePath)) throw new Error(`${label} not found: ${filePath}`);
  const report = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (!Array.isArray(report.cases)) throw new Error(`${label} must contain a cases array`);
  return report;
}

async function run({ github, context, core, env = process.env }) {
  const current = readReport(env.REPORT_PATH, 'Evaluation report');
  const baseline = readReport(env.BASELINE_PATH, 'Baseline report');
  const thresholds = {
    tokens: Number(env.TOKEN_REGRESSION_PERCENT || 20),
    cost: Number(env.COST_REGRESSION_PERCENT || 20),
    latency: Number(env.LATENCY_REGRESSION_PERCENT || 20),
  };
  const result = await postOrUpdateComment({ github, context, current, baseline, thresholds });
  if (result.skipped) core.info('No pull request context; skipping AgentEval comment.');
  else core.info(`AgentEval PR comment ${result.updated ? 'updated' : 'created'} (${result.commentId}).`);
  return result;
}

module.exports = { buildComment, postOrUpdateComment, readReport, run };
