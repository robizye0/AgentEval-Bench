const test = require('node:test');
const assert = require('node:assert/strict');
const { buildComment, postOrUpdateComment } = require('../comment.js');

const current = {
  passed: true,
  average_deterministic_accuracy: 1,
  cases: [{
    id: 'search', passed: true, deterministic_accuracy_score: 1, total_cost_usd: 0.02,
    details: { token_count: 150, latency_ms: 700 },
  }],
};
const baseline = {
  passed: true,
  average_deterministic_accuracy: 0.9,
  cases: [{
    id: 'search', passed: true, deterministic_accuracy_score: 0.9, total_cost_usd: 0.01,
    details: { token_count: 100, latency_ms: 500 },
  }],
};

test('comment contains per-case status, accuracy delta, and regression alerts', () => {
  const comment = buildComment(current, baseline, { tokens: 20, cost: 20, latency: 20 });
  assert.match(comment, /search/);
  assert.match(comment, /PASS/);
  assert.match(comment, /\+10\.0 pp/);
  assert.match(comment, /tokens \+50%/);
  assert.match(comment, /cost \+100%/);
  assert.match(comment, /latency \+40%/);
});

test('existing bot comment is updated rather than duplicated', async () => {
  const calls = [];
  const github = {
    paginate: async () => [{ id: 7, body: '<!-- agenteval-bot -->\nold' }],
    rest: { issues: {
      updateComment: async (payload) => calls.push(['update', payload]),
      createComment: async (payload) => calls.push(['create', payload]),
    } },
  };
  await postOrUpdateComment({ github, context: { issue: { number: 3 }, repo: { owner: 'o', repo: 'r' } }, current, baseline });
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], 'update');
  assert.equal(calls[0][1].comment_id, 7);
});
