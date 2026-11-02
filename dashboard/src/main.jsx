import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const initialApi = import.meta.env.VITE_AGENTEVAL_API_URL || 'http://localhost:8000';

function App() {
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem('agenteval-api') || initialApi);
  const [token, setToken] = useState(() => sessionStorage.getItem('agenteval-token') || '');
  const [activeApiUrl, setActiveApiUrl] = useState(() => localStorage.getItem('agenteval-api') || initialApi);
  const [activeToken, setActiveToken] = useState(() => sessionStorage.getItem('agenteval-token') || '');
  const [cases, setCases] = useState([]);
  const [connection, setConnection] = useState('connecting');
  const [error, setError] = useState('');
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selected, setSelected] = useState(null);

  const headers = useMemo(() => activeToken ? { Authorization: `Bearer ${activeToken}` } : {}, [activeToken]);
  const base = activeApiUrl.trim().replace(/\/$/, '');

  useEffect(() => {
    localStorage.setItem('agenteval-api', activeApiUrl);
    sessionStorage.setItem('agenteval-token', activeToken);
    const controller = new AbortController();
    let stopped = false;
    let retryTimer;

    const loadSnapshot = async () => {
      const response = await fetch(`${base}/eval/dashboard`, { headers, signal: controller.signal });
      if (!response.ok) throw new Error(`Dashboard request failed (${response.status})`);
      const payload = await response.json();
      setCases(payload.cases || []);
      setLastUpdate(new Date());
    };

    const stream = async () => {
      try {
        await loadSnapshot();
        while (!stopped) {
          setConnection('live');
          const response = await fetch(`${base}/eval/events`, { headers, signal: controller.signal });
          if (!response.ok || !response.body) throw new Error(`Event stream failed (${response.status})`);
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          while (!stopped) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true }).replaceAll('\r\n', '\n');
            const frames = buffer.split('\n\n');
            buffer = frames.pop() || '';
            for (const frame of frames) {
              const eventType = frame.split('\n').find((line) => line.startsWith('event:'))?.slice(6).trim();
              const data = frame.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
              if (!data) continue;
              const payload = JSON.parse(data);
              if (eventType === 'snapshot') setCases(payload.cases || []);
              else if (eventType === 'evaluation') {
                setCases((previous) => {
                  const next = previous.filter((item) => item.case_id !== payload.case_id);
                  next.push({
                    case_id: payload.case_id,
                    status: payload.status,
                    step_count: payload.step_count,
                    final: payload.final,
                    passed: payload.result?.passed,
                    token_count: payload.token_count,
                    cost_usd: payload.cost_usd,
                    latency_ms: payload.latency_ms,
                    result: payload.result,
                  });
                  return next.sort((a, b) => a.case_id.localeCompare(b.case_id));
                });
              }
              setLastUpdate(new Date());
            }
          }
          if (!stopped) throw new Error('Event stream disconnected; reconnecting');
        }
      } catch (err) {
        if (stopped || err.name === 'AbortError') return;
        setConnection('reconnecting');
        setError(err.message || 'Could not connect to the AgentEval API');
        retryTimer = setTimeout(stream, 2000);
      }
    };

    setConnection('connecting');
    setError('');
    stream();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(retryTimer);
    };
  }, [base, headers]);

  const total = cases.length;
  const passed = cases.filter((item) => item.passed === true).length;
  const running = cases.filter((item) => !item.final).length;
  const tokens = cases.reduce((sum, item) => sum + (item.token_count || 0), 0);
  const cost = cases.reduce((sum, item) => sum + (item.cost_usd || 0), 0);
  const avgLatency = total ? cases.reduce((sum, item) => sum + (item.latency_ms || 0), 0) / total : 0;

  return <main className="shell">
    <header className="topbar">
      <a className="brand" href="#"><span className="brand-mark">A</span><span>AgentEval<span className="brand-muted"> / CONTROL ROOM</span></span></a>
      <div className="top-right"><span className={`live-dot ${connection}`} /> <span className="connection-label">{connection === 'live' ? 'LIVE STREAM' : connection.toUpperCase()}</span><span className="divider" /> <span className="clock">{lastUpdate ? `Updated ${lastUpdate.toLocaleTimeString()}` : 'Waiting for telemetry'}</span></div>
    </header>

    <section className="intro">
      <div><p className="eyebrow">EVALUATION TELEMETRY <span>·</span> REAL-TIME</p><h1>Agent control room</h1><p className="subtitle">Watch trajectories, quality gates, and spend as they happen.</p></div>
      <div className="api-config"><label htmlFor="api">API ENDPOINT <span>·</span> BEARER TOKEN (OPTIONAL)</label><div className="api-row"><input id="api" value={apiUrl} onChange={(event) => setApiUrl(event.target.value)} spellCheck="false" aria-label="API endpoint" /><input id="token" type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="Token" aria-label="Bearer token" /><button onClick={() => { setActiveApiUrl(apiUrl); setActiveToken(token); setConnection('connecting'); setError(''); }}>CONNECT <span>↗</span></button></div></div>
    </section>

    <section className="metrics" aria-label="Evaluation summary">
      <Metric label="Tracked cases" value={total.toString().padStart(2, '0')} hint={`${running} in progress`} icon="◉" />
      <Metric label="Pass rate" value={total ? `${Math.round((passed / total) * 100)}%` : '—'} hint={`${passed} passed · ${total - passed} failed`} icon="↗" accent />
      <Metric label="Tokens consumed" value={tokens.toLocaleString()} hint="Across active cases" icon="⌘" />
      <Metric label="Total cost" value={`$${cost.toFixed(4)}`} hint="Cumulative run spend" icon="$" />
      <Metric label="Mean latency" value={`${Math.round(avgLatency)} ms`} hint="Per trajectory" icon="◷" />
    </section>

    <section className="panel">
      <div className="panel-head"><div><div className="panel-title">Live trajectories <span className="count">{String(total).padStart(2, '0')}</span></div><div className="panel-sub">Case-level execution and regression signals</div></div><span className="stream-badge"><i /> STREAMING</span></div>
      {error && <div className="error-banner"><span>Connection issue</span> {error} <small>Retrying automatically…</small></div>}
      <div className="table-wrap"><table><thead><tr><th>CASE</th><th>STATUS</th><th>STEPS</th><th>TOKENS</th><th>COST</th><th>LATENCY</th><th>QUALITY</th><th /></tr></thead>
        <tbody>{cases.map((item) => <tr key={item.case_id} onClick={() => setSelected(selected === item.case_id ? null : item.case_id)} className={selected === item.case_id ? 'selected' : ''}>
          <td><span className="case-icon">{item.case_id.slice(0, 1).toUpperCase()}</span><span className="case-id">{item.case_id}</span></td>
          <td><Status item={item} /></td><td className="mono">{item.step_count ?? 0}<span className="unit"> steps</span></td>
          <td className="mono">{(item.token_count || 0).toLocaleString()}</td><td className="mono">${(item.cost_usd || 0).toFixed(4)}</td>
          <td className="mono">{Math.round(item.latency_ms || 0)}<span className="unit"> ms</span></td>
          <td><div className="quality"><span><i style={{ width: `${Math.max(0, Math.min(100, (item.result?.tool_accuracy_score || 0) * 100))}%` }} /></span><b>{item.result ? `${Math.round((item.result.tool_accuracy_score || 0) * 100)}%` : '—'}</b></div></td><td className="chevron">{selected === item.case_id ? '−' : '+'}</td>
        </tr>)}
        {!total && <tr><td colSpan="8" className="empty"><div className="empty-mark">◌</div><strong>No trajectories yet</strong><span>Start sending events to <code>/eval/ingest</code> to see live activity here.</span></td></tr>}
        </tbody></table></div>
      {selected && cases.find((item) => item.case_id === selected)?.result && <pre className="details">{JSON.stringify(cases.find((item) => item.case_id === selected).result, null, 2)}</pre>}
      <div className="panel-foot"><span><i className="foot-dot" /> Updates arrive over server-sent events</span><span>Protected with optional bearer authentication</span></div>
    </section>
    <footer><span>AGENTEVAL-BENCH <b>v0.1</b></span><span>Evaluation infrastructure for agents</span></footer>
  </main>;
}

function Metric({ label, value, hint, icon, accent }) {
  return <article className={`metric ${accent ? 'metric-accent' : ''}`}><div className="metric-top"><span>{label}</span><span className="metric-icon">{icon}</span></div><div className="metric-value">{value}</div><div className="metric-hint">{hint}</div></article>;
}

function Status({ item }) {
  const type = item.final ? (item.passed ? 'passed' : 'failed') : 'running';
  const label = type === 'passed' ? 'PASSED' : type === 'failed' ? 'FAILED' : 'RUNNING';
  return <span className={`status ${type}`}><i />{label}</span>;
}

createRoot(document.getElementById('root')).render(<App />);
